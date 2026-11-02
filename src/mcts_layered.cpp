#include "brogameagent/mcts.h"
#include "mcts_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace brogameagent::mcts {

// ─── OptionMcts ────────────────────────────────────────────────────────────

namespace {

OptionMcts::ONode* omcts_best_uct_child(OptionMcts::ONode* node, float c) {
    OptionMcts::ONode* best = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();
    const float ln_n = std::log(static_cast<float>(std::max(1, node->visits)));
    for (auto& up : node->children) {
        auto* ch = up.get();
        if (ch->visits == 0) return ch;
        float exploit = ch->mean();
        float explore = c * std::sqrt(ln_n / static_cast<float>(ch->visits));
        float score = exploit + explore;
        if (score > best_score) { best_score = score; best = ch; }
    }
    return best;
}

int omcts_count_tree(const OptionMcts::ONode* n) {
    if (!n) return 0;
    int total = 1;
    for (const auto& ch : n->children) total += omcts_count_tree(ch.get());
    return total;
}

// Shared hero + others stepping helper for single-hero option execution.
// Advances the world by one decision window (action_repeat sim ticks) with
// `hero_action` driving the hero and `opponent` driving every other agent.
// Identical semantics to Mcts::step_decision_, factored here so both
// in-tree execution and live execute_option() use the same code path.
void step_hero_window(World& world, Agent& hero,
                      const CombatAction& hero_action,
                      const OpponentPolicy& opponent,
                      float dt, int action_repeat) {
    std::vector<std::pair<Agent*, CombatAction>> others;
    others.reserve(world.agents().size());
    for (Agent* a : world.agents()) {
        if (a == &hero) continue;
        CombatAction act{};
        if (a->unit().alive() && opponent) act = opponent(*a, world);
        others.emplace_back(a, act);
    }
    for (int t = 0; t < action_repeat; t++) {
        mcts::apply(hero, world, hero_action, dt);
        for (auto& [a, act] : others) mcts::apply(*a, world, act, dt);
        world.stepProjectiles(dt);
        world.cullProjectiles();
        if (is_terminal_for(hero, world)) break;
    }
}

} // namespace

std::vector<const Option*> OptionMcts::legal_options_(
        const Agent& hero, const World& world) const {
    std::vector<const Option*> out;
    out.reserve(options_.size());
    for (const auto& sp : options_) {
        if (sp && sp->can_initiate(hero, world)) out.push_back(sp.get());
    }
    return out;
}

void OptionMcts::step_option_(World& world, Agent& hero, const Option& option) {
    const float dt = cfg_.sim_dt;
    for (int w = 0; w < cfg_.option_max_windows; w++) {
        if (is_terminal_for(hero, world)) return;
        CombatAction hero_action = option.step(hero, world, w);
        step_hero_window(world, hero, hero_action, opponent_, dt, cfg_.action_repeat);
        if (is_terminal_for(hero, world)) return;
        if (option.should_terminate(hero, world, w + 1)) return;
    }
}

int OptionMcts::execute_option(World& world, Agent& hero, const Option& option) {
    const float dt = cfg_.sim_dt;
    for (int w = 0; w < cfg_.option_max_windows; w++) {
        if (is_terminal_for(hero, world)) return w;
        CombatAction hero_action = option.step(hero, world, w);
        step_hero_window(world, hero, hero_action, opponent_, dt, cfg_.action_repeat);
        if (is_terminal_for(hero, world)) return w + 1;
        if (option.should_terminate(hero, world, w + 1)) return w + 1;
    }
    return cfg_.option_max_windows;
}

OptionMcts::ONode* OptionMcts::select_(ONode* node, World& world, Agent& hero) {
    while (node->fully_expanded() && !node->is_leaf()) {
        if (is_terminal_for(hero, world)) return node;
        ONode* next = omcts_best_uct_child(node, cfg_.uct_c);
        if (!next) return node;
        step_option_(world, hero, *next->action);
        node = next;
    }
    return node;
}

OptionMcts::ONode* OptionMcts::expand_(ONode* node, World& world, Agent& hero) {
    if (is_terminal_for(hero, world) || node->untried.empty()) return node;
    const Option* opt = node->untried.back();
    node->untried.pop_back();
    step_option_(world, hero, *opt);

    auto child = std::make_unique<ONode>();
    child->action = opt;
    child->parent = node;
    child->untried = legal_options_(hero, world);
    ONode* raw = child.get();
    node->children.push_back(std::move(child));
    return raw;
}

float OptionMcts::rollout_(World& world, Agent& hero) {
    const int hero_id = hero.unit().id;
    auto eval = [&]() {
        return evaluator_
            ? evaluator_->evaluate(world, hero_id)
            : HpDeltaEvaluator{}.evaluate(world, hero_id);
    };
    if (cfg_.use_leaf_value) return eval();

    int steps = 0;
    while (steps < cfg_.rollout_horizon && !is_terminal_for(hero, world)) {
        auto opts = legal_options_(hero, world);
        if (opts.empty()) break;
        int i = world.randInt(0, static_cast<int>(opts.size()) - 1);
        step_option_(world, hero, *opts[static_cast<size_t>(i)]);
        steps++;
    }
    return eval();
}

const Option* OptionMcts::search(World& world, Agent& hero) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    WorldSnapshot saved = world.snapshot();

    bool reused = false;
    if (root_) {
        reused = true;
    } else {
        root_ = std::make_unique<ONode>();
        root_->untried = legal_options_(hero, world);
    }

    if (root_->untried.empty() && root_->children.empty()) {
        // No option can initiate from this state — caller must fall back.
        stats_ = SearchStats{};
        stats_.reused_root = reused;
        world.restore(saved);
        return nullptr;
    }

    const int  iter_cap = cfg_.iterations > 0 ? cfg_.iterations
                                              : std::numeric_limits<int>::max();
    const bool time_cap = cfg_.budget_ms > 0;

    int it = 0;
    for (; it < iter_cap; it++) {
        if (time_cap) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - t_start).count();
            if (elapsed >= cfg_.budget_ms) break;
        }
        world.restore(saved);
        world.seed(cfg_.seed
                   + static_cast<uint64_t>(it)
                   + static_cast<uint64_t>(root_->visits) * 7919ull);

        ONode* leaf  = select_(root_.get(), world, hero);
        ONode* child = expand_(leaf, world, hero);
        float  value = rollout_(world, hero);

        ONode* n = child;
        while (n) { n->visits++; n->total_value += value; n = n->parent; }
    }

    world.restore(saved);

    ONode* best = nullptr;
    int best_visits = -1;
    for (auto& up : root_->children) {
        if (up->visits > best_visits) { best_visits = up->visits; best = up.get(); }
    }

    const auto t_end = clock::now();
    stats_.iterations    = it;
    stats_.root_children = static_cast<int>(root_->children.size());
    stats_.tree_size     = omcts_count_tree(root_.get());
    stats_.best_mean     = best ? best->mean() : 0.0f;
    stats_.best_visits   = best ? best->visits : 0;
    stats_.elapsed_ms    = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
    stats_.reused_root   = reused;

    return best ? best->action : nullptr;
}

void OptionMcts::advance_root(const Option* committed) {
    if (!root_ || !committed) { root_.reset(); return; }
    std::unique_ptr<ONode> promoted;
    for (auto& up : root_->children) {
        // Match by name (stable identity) rather than pointer, so a caller
        // that rebuilds its option set between decisions doesn't invalidate
        // the retained tree.
        if (up->action && up->action->name() == committed->name()) {
            promoted = std::move(up);
            break;
        }
    }
    if (promoted) { promoted->parent = nullptr; root_ = std::move(promoted); }
    else          { root_.reset(); }
}


// ─── TeamOptionMcts ───────────────────────────────────────────────────────

namespace {

TeamOptionMcts::TNode* tomcts_best_uct_child(TeamOptionMcts::TNode* node, float c) {
    TeamOptionMcts::TNode* best = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();
    const float ln_n = std::log(static_cast<float>(std::max(1, node->visits)));
    for (auto& up : node->children) {
        auto* ch = up.get();
        if (ch->visits == 0) return ch;
        float exploit = ch->mean();
        float explore = c * std::sqrt(ln_n / static_cast<float>(ch->visits));
        float score = exploit + explore;
        if (score > best_score) { best_score = score; best = ch; }
    }
    return best;
}

int tomcts_count_tree(const TeamOptionMcts::TNode* n) {
    if (!n) return 0;
    int total = 1;
    for (const auto& ch : n->children) total += tomcts_count_tree(ch.get());
    return total;
}

} // namespace

std::vector<const TeamOption*> TeamOptionMcts::legal_options_(
        const std::vector<Agent*>& heroes, const World& world) const {
    std::vector<const TeamOption*> out;
    out.reserve(options_.size());
    for (const auto& sp : options_) {
        if (sp && sp->can_initiate(heroes, world)) out.push_back(sp.get());
    }
    return out;
}

void TeamOptionMcts::step_option_(World& world, const std::vector<Agent*>& heroes,
                                    const TeamOption& option) {
    if (heroes.empty()) return;
    const int team_id = heroes.front()->unit().teamId;
    for (int w = 0; w < cfg_.option_max_windows; w++) {
        if (team_terminal(world, team_id)) return;
        auto hero_actions = option.step(heroes, world, w);
        if (hero_actions.size() != heroes.size()) {
            hero_actions.assign(heroes.size(), CombatAction{});
        }
        step_team_window(world, heroes, hero_actions, opponent_,
                         cfg_.sim_dt, cfg_.action_repeat, team_id);
        if (team_terminal(world, team_id)) return;
        if (option.should_terminate(heroes, world, w + 1)) return;
    }
}

int TeamOptionMcts::execute_option(World& world, const std::vector<Agent*>& heroes,
                                    const TeamOption& option) {
    if (heroes.empty()) return 0;
    const int team_id = heroes.front()->unit().teamId;
    for (int w = 0; w < cfg_.option_max_windows; w++) {
        if (team_terminal(world, team_id)) return w;
        auto hero_actions = option.step(heroes, world, w);
        if (hero_actions.size() != heroes.size()) {
            hero_actions.assign(heroes.size(), CombatAction{});
        }
        step_team_window(world, heroes, hero_actions, opponent_,
                         cfg_.sim_dt, cfg_.action_repeat, team_id);
        if (team_terminal(world, team_id)) return w + 1;
        if (option.should_terminate(heroes, world, w + 1)) return w + 1;
    }
    return cfg_.option_max_windows;
}

TeamOptionMcts::TNode* TeamOptionMcts::select_(TNode* node, World& world,
                                                const std::vector<Agent*>& heroes) {
    const int team_id = heroes.front()->unit().teamId;
    while (node->fully_expanded() && !node->is_leaf()) {
        if (team_terminal(world, team_id)) return node;
        TNode* next = tomcts_best_uct_child(node, cfg_.uct_c);
        if (!next) return node;
        step_option_(world, heroes, *next->action);
        node = next;
    }
    return node;
}

TeamOptionMcts::TNode* TeamOptionMcts::expand_(TNode* node, World& world,
                                                const std::vector<Agent*>& heroes) {
    const int team_id = heroes.front()->unit().teamId;
    if (team_terminal(world, team_id) || node->untried.empty()) return node;
    const TeamOption* opt = node->untried.back();
    node->untried.pop_back();
    step_option_(world, heroes, *opt);

    auto child = std::make_unique<TNode>();
    child->action = opt;
    child->parent = node;
    child->untried = legal_options_(heroes, world);
    TNode* raw = child.get();
    node->children.push_back(std::move(child));
    return raw;
}

float TeamOptionMcts::rollout_(World& world, const std::vector<Agent*>& heroes, int team_id) {
    auto eval = [&]() {
        return evaluator_
            ? evaluator_->evaluate(world, team_id)
            : TeamHpDeltaEvaluator{}.evaluate(world, team_id);
    };
    if (cfg_.use_leaf_value) return eval();

    int steps = 0;
    while (steps < cfg_.rollout_horizon && !team_terminal(world, team_id)) {
        auto opts = legal_options_(heroes, world);
        if (opts.empty()) break;
        int i = world.randInt(0, static_cast<int>(opts.size()) - 1);
        step_option_(world, heroes, *opts[static_cast<size_t>(i)]);
        steps++;
    }
    return eval();
}

const TeamOption* TeamOptionMcts::search(World& world, const std::vector<Agent*>& heroes) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    if (heroes.empty()) return nullptr;
    const int team_id = heroes.front()->unit().teamId;

    WorldSnapshot saved = world.snapshot();

    bool reused = false;
    if (root_) {
        reused = true;
    } else {
        root_ = std::make_unique<TNode>();
        root_->untried = legal_options_(heroes, world);
    }

    if (root_->untried.empty() && root_->children.empty()) {
        stats_ = SearchStats{};
        stats_.reused_root = reused;
        world.restore(saved);
        return nullptr;
    }

    const int  iter_cap = cfg_.iterations > 0 ? cfg_.iterations
                                              : std::numeric_limits<int>::max();
    const bool time_cap = cfg_.budget_ms > 0;

    int it = 0;
    for (; it < iter_cap; it++) {
        if (time_cap) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - t_start).count();
            if (elapsed >= cfg_.budget_ms) break;
        }
        world.restore(saved);
        world.seed(cfg_.seed
                   + static_cast<uint64_t>(it)
                   + static_cast<uint64_t>(root_->visits) * 7919ull);

        TNode* leaf  = select_(root_.get(), world, heroes);
        TNode* child = expand_(leaf, world, heroes);
        float  value = rollout_(world, heroes, team_id);

        TNode* n = child;
        while (n) { n->visits++; n->total_value += value; n = n->parent; }
    }

    world.restore(saved);

    TNode* best = nullptr;
    int best_visits = -1;
    for (auto& up : root_->children) {
        if (up->visits > best_visits) { best_visits = up->visits; best = up.get(); }
    }

    const auto t_end = clock::now();
    stats_.iterations    = it;
    stats_.root_children = static_cast<int>(root_->children.size());
    stats_.tree_size     = tomcts_count_tree(root_.get());
    stats_.best_mean     = best ? best->mean() : 0.0f;
    stats_.best_visits   = best ? best->visits : 0;
    stats_.elapsed_ms    = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
    stats_.reused_root   = reused;

    return best ? best->action : nullptr;
}

void TeamOptionMcts::advance_root(const TeamOption* committed) {
    if (!root_ || !committed) { root_.reset(); return; }
    std::unique_ptr<TNode> promoted;
    for (auto& up : root_->children) {
        if (up->action && up->action->name() == committed->name()) {
            promoted = std::move(up);
            break;
        }
    }
    if (promoted) { promoted->parent = nullptr; root_ = std::move(promoted); }
    else          { root_.reset(); }
}


// ─── Commander ─────────────────────────────────────────────────────────────

void Commander::reset() {
    assignments_.clear();
    hero_states_.clear();
    windows_left_ = 0;
}

std::string Commander::committed_option_for_hero(size_t hero_idx) const {
    if (hero_idx >= hero_states_.size()) return {};
    const Option* opt = hero_states_[hero_idx].current_option;
    return opt ? opt->name() : std::string{};
}

std::vector<CombatAction> Commander::decide(World& world,
                                              const std::vector<Agent*>& heroes) {
    std::vector<CombatAction> out(heroes.size());
    if (heroes.empty() || roles_.empty()) return out;

    // Resize per-hero state to match roster. If the team composition changed,
    // engines are dropped — they're keyed to a role's option set which may
    // no longer be valid.
    if (hero_states_.size() != heroes.size()) {
        hero_states_.clear();
        hero_states_.resize(heroes.size());
        windows_left_ = 0;
    }

    // Re-assign roles when the window elapses. On re-assignment we reset
    // each hero's current option — a reassigned hero's previous option may
    // not belong to their new role's set.
    const bool need_replan = (windows_left_ <= 0) ||
                             (assignments_.size() != heroes.size());
    if (need_replan) {
        if (assigner_) {
            assignments_ = assigner_(heroes, world);
        } else {
            // Default: round-robin heroes across registered roles.
            assignments_.assign(heroes.size(), 0);
            for (size_t i = 0; i < heroes.size(); i++) {
                assignments_[i] = static_cast<int>(i % roles_.size());
            }
        }
        if (assignments_.size() != heroes.size()) assignments_.resize(heroes.size(), 0);

        // Dropping options on reassign is the common-case behaviour; a hero
        // moving from "lead" to "support" shouldn't finish their old push.
        for (auto& hs : hero_states_) {
            hs.current_option = nullptr;
            hs.ticks_in_option = 0;
            hs.engine.reset();
        }
        windows_left_ = cfg_.replan_every_windows;
    }

    for (size_t i = 0; i < heroes.size(); i++) {
        Agent* h = heroes[i];
        if (!h || !h->unit().alive()) continue;
        int r = assignments_[i];
        if (r < 0 || r >= static_cast<int>(roles_.size())) continue;
        const Role& role = roles_[r];
        if (role.options.empty()) continue;

        HeroState& hs = hero_states_[i];
        if (!hs.engine) {
            hs.engine = std::make_unique<OptionMcts>(cfg_.role_cfg);
            hs.engine->set_options(role.options);
            auto ev = role.evaluator ? role.evaluator : default_eval_;
            if (ev) hs.engine->set_evaluator(ev);
            if (opponent_) hs.engine->set_opponent_policy(opponent_);
        }

        // Pick a new option when none is active or the current one is
        // terminated. Search is only invoked here — steady-state stepping
        // just calls option.step() which is cheap.
        if (!hs.current_option ||
            hs.current_option->should_terminate(*h, world, hs.ticks_in_option) ||
            hs.ticks_in_option >= cfg_.role_cfg.option_max_windows) {
            const Option* picked = hs.engine->search(world, *h);
            hs.current_option = picked;
            hs.ticks_in_option = 0;
            hs.engine->advance_root(picked);
        }

        if (hs.current_option) {
            out[i] = hs.current_option->step(*h, world, hs.ticks_in_option);
            hs.ticks_in_option++;
        }
    }

    windows_left_--;
    return out;
}

// ─── TacticPrior ───────────────────────────────────────────────────────────

std::vector<float> TacticPrior::score(
    const Agent& self, const World& world,
    const std::vector<CombatAction>& actions) const
{
    std::vector<float> w(actions.size(), other_weight_);
    if (!self.unit().alive() || actions.empty()) return w;
    CombatAction target = tactic_to_action(tactic_, self, world);
    for (size_t i = 0; i < actions.size(); i++) {
        if (actions[i] == target) w[i] = match_weight_;
    }
    return w;
}


// ─── LayeredPlanner ────────────────────────────────────────────────────────

void LayeredPlanner::reset() {
    tactic_mcts_.reset_tree();
    fine_mcts_.reset_tree();
    committed_tactic_ = Tactic{};
    windows_left_ = 0;
    stats_ = Stats{};
}

TeamMcts::JointAction LayeredPlanner::decide(
    World& world, const std::vector<Agent*>& heroes)
{
    stats_.replanned_this_call = false;

    // Configure the tactic layer lazily on first use (or on any config change
    // — cheap to re-apply every call since setters just move shared_ptrs).
    tactic_mcts_.set_config(cfg_.tactic_cfg);
    if (evaluator_)      tactic_mcts_.set_evaluator(evaluator_);
    if (opponent_)       tactic_mcts_.set_opponent_policy(opponent_);

    fine_mcts_.set_config(cfg_.fine_cfg);
    if (evaluator_)      fine_mcts_.set_evaluator(evaluator_);
    if (rollout_policy_) fine_mcts_.set_rollout_policy(rollout_policy_);
    if (opponent_)       fine_mcts_.set_opponent_policy(opponent_);

    // (Re)plan the tactic when the current commitment has expired. Tree is
    // reset each time because world state has moved on between calls and the
    // prior tree's stats were collected for a stale snapshot.
    if (windows_left_ <= 0) {
        tactic_mcts_.reset_tree();
        committed_tactic_ = tactic_mcts_.search(world, heroes);
        stats_.tactic_stats = tactic_mcts_.last_stats();
        windows_left_ = std::max(1, cfg_.tactic_cfg.tactic_window_decisions);
        stats_.replanned_this_call = true;
    }

    // Bias the fine search toward the committed tactic via TacticPrior.
    if (!prior_) prior_ = std::make_shared<TacticPrior>();
    prior_->set_tactic(committed_tactic_);
    prior_->set_match_weight(cfg_.tactic_match_weight);
    prior_->set_other_weight(cfg_.tactic_other_weight);
    fine_mcts_.set_prior(prior_);

    fine_mcts_.reset_tree();
    TeamMcts::JointAction joint = fine_mcts_.search(world, heroes);
    stats_.fine_stats = fine_mcts_.last_stats();

    windows_left_--;
    stats_.committed_tactic      = committed_tactic_;
    stats_.windows_until_replan  = windows_left_;
    return joint;
}

} // namespace brogameagent::mcts
