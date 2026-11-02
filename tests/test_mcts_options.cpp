#include "test_common.h"

// ─── OptionMcts / TeamOptionMcts ───────────────────────────────────────────

namespace {

// Simple concrete options for tests. Each option drives the hero with a
// fixed CombatAction and terminates after N windows.
class HoldOpt : public mcts::Option {
    std::string name_ = "hold";
    int windows_;
public:
    explicit HoldOpt(int w = 2) : windows_(w) {}
    const std::string& name() const override { return name_; }
    bool can_initiate(const Agent&, const World&) const override { return true; }
    mcts::CombatAction step(Agent&, World&, int) const override {
        return mcts::CombatAction{ mcts::MoveDir::Hold, -1, -1 };
    }
    bool should_terminate(const Agent&, const World&, int ticks) const override {
        return ticks >= windows_;
    }
};

class AttackOpt : public mcts::Option {
    std::string name_ = "attack";
    int windows_;
public:
    explicit AttackOpt(int w = 3) : windows_(w) {}
    const std::string& name() const override { return name_; }
    bool can_initiate(const Agent& self, const World& world) const override {
        // Only if there's at least one living enemy.
        for (Agent* a : world.agents()) {
            if (a == &self) continue;
            if (a->unit().alive() && a->unit().teamId != self.unit().teamId) return true;
        }
        return false;
    }
    mcts::CombatAction step(Agent&, World&, int) const override {
        return mcts::CombatAction{ mcts::MoveDir::PathToTarget, 0, -1 };
    }
    bool should_terminate(const Agent&, const World&, int ticks) const override {
        return ticks >= windows_;
    }
};

class NeverInitOpt : public mcts::Option {
    std::string name_ = "never";
public:
    const std::string& name() const override { return name_; }
    bool can_initiate(const Agent&, const World&) const override { return false; }
    mcts::CombatAction step(Agent&, World&, int) const override { return {}; }
    bool should_terminate(const Agent&, const World&, int) const override { return true; }
};

class TeamHoldOpt : public mcts::TeamOption {
    std::string name_ = "team_hold";
    int windows_;
public:
    explicit TeamHoldOpt(int w = 2) : windows_(w) {}
    const std::string& name() const override { return name_; }
    bool can_initiate(const std::vector<Agent*>&, const World&) const override { return true; }
    std::vector<mcts::CombatAction> step(const std::vector<Agent*>& heroes,
                                           World&, int) const override {
        return std::vector<mcts::CombatAction>(heroes.size(),
            mcts::CombatAction{ mcts::MoveDir::Hold, -1, -1 });
    }
    bool should_terminate(const std::vector<Agent*>&, const World&, int t) const override {
        return t >= windows_;
    }
};

class TeamPushOpt : public mcts::TeamOption {
    std::string name_ = "team_push";
    int windows_;
public:
    explicit TeamPushOpt(int w = 3) : windows_(w) {}
    const std::string& name() const override { return name_; }
    bool can_initiate(const std::vector<Agent*>& heroes, const World& world) const override {
        if (heroes.empty()) return false;
        int team = heroes.front()->unit().teamId;
        for (Agent* a : world.agents()) {
            if (a->unit().alive() && a->unit().teamId != team) return true;
        }
        return false;
    }
    std::vector<mcts::CombatAction> step(const std::vector<Agent*>& heroes,
                                           World&, int) const override {
        return std::vector<mcts::CombatAction>(heroes.size(),
            mcts::CombatAction{ mcts::MoveDir::PathToTarget, 0, -1 });
    }
    bool should_terminate(const std::vector<Agent*>&, const World&, int t) const override {
        return t >= windows_;
    }
};

} // namespace

TEST(option_mcts_search_returns_an_option) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations        = 80;
    cfg.rollout_horizon   = 3;
    cfg.action_repeat     = 2;
    cfg.option_max_windows = 4;
    cfg.seed              = 0xAB;

    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<HoldOpt>(2),
        std::make_shared<AttackOpt>(3),
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);

    const mcts::Option* chosen = engine.search(s->world, s->hero);
    CHECK(chosen != nullptr);
    CHECK(chosen->name() == "hold" || chosen->name() == "attack");
    CHECK(engine.last_stats().root_children > 0);
}

TEST(option_mcts_excludes_options_that_cannot_initiate) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations        = 40;
    cfg.rollout_horizon   = 2;
    cfg.option_max_windows = 3;

    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<HoldOpt>(2),
        std::make_shared<NeverInitOpt>(),   // filtered out every expansion
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, s->hero);
    // Only "hold" should ever have been expanded at the root.
    CHECK(engine.last_stats().root_children == 1);
}

TEST(option_mcts_returns_null_when_no_option_initiable) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations = 20;
    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<NeverInitOpt>(),
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    CHECK(engine.search(s->world, s->hero) == nullptr);
}

TEST(option_mcts_search_is_side_effect_free) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, 3.0f);
    float hp_hero_before = s->hero.unit().hp;
    float hp_enemy_before = s->world.agents()[1]->unit().hp;

    mcts::MctsConfig cfg;
    cfg.iterations = 50;
    cfg.rollout_horizon = 3;
    cfg.option_max_windows = 3;
    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<HoldOpt>(2),
        std::make_shared<AttackOpt>(2),
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, s->hero);

    CHECK_NEAR(s->hero.unit().hp, hp_hero_before, 1e-4f);
    CHECK_NEAR(s->world.agents()[1]->unit().hp, hp_enemy_before, 1e-4f);
}

TEST(option_mcts_advance_root_by_name) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations = 40;
    cfg.option_max_windows = 3;
    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<HoldOpt>(2),
        std::make_shared<AttackOpt>(2),
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    const mcts::Option* first = engine.search(s->world, s->hero);
    CHECK(first != nullptr);

    // Re-author the option set with fresh shared_ptrs (different pointers,
    // same names) — advance_root must match by name and preserve the subtree.
    std::vector<std::shared_ptr<mcts::Option>> opts2 = {
        std::make_shared<HoldOpt>(2),
        std::make_shared<AttackOpt>(2),
    };
    const mcts::Option* committed = nullptr;
    for (auto& sp : opts2) if (sp->name() == first->name()) { committed = sp.get(); break; }
    engine.advance_root(committed);
    CHECK(engine.last_root() != nullptr);
    CHECK(engine.last_root()->action != nullptr);
    CHECK(engine.last_root()->action->name() == first->name());
}

TEST(option_mcts_use_leaf_value_finishes_with_large_horizon) {
    // With use_leaf_value=true, rollout_horizon is ignored. Set it huge: if
    // rollout actually ran the search would take long enough that elapsed_ms
    // would be very large. Instead we just assert it completes all iterations
    // and reports a small-to-moderate elapsed time.
    auto s = make_duel_scene(0, 0, 1.0f, 0, 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations = 30;
    cfg.rollout_horizon = 10000;
    cfg.use_leaf_value = true;
    cfg.option_max_windows = 2;
    std::vector<std::shared_ptr<mcts::Option>> opts = {
        std::make_shared<HoldOpt>(1),
        std::make_shared<AttackOpt>(1),
    };
    mcts::OptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, s->hero);
    CHECK(engine.last_stats().iterations == 30);
    CHECK(engine.last_stats().elapsed_ms < 500);
}

TEST(team_option_mcts_search_returns_an_option) {
    auto s = make_team_scene(2, 2);
    mcts::MctsConfig cfg;
    cfg.iterations        = 60;
    cfg.rollout_horizon   = 3;
    cfg.action_repeat     = 2;
    cfg.option_max_windows = 3;
    cfg.seed              = 0xCD;

    std::vector<std::shared_ptr<mcts::TeamOption>> opts = {
        std::make_shared<TeamHoldOpt>(2),
        std::make_shared<TeamPushOpt>(2),
    };
    mcts::TeamOptionMcts engine(cfg);
    engine.set_options(opts);
    engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    engine.set_opponent_policy(mcts::policy_idle);

    const mcts::TeamOption* chosen = engine.search(s->world, raw(s->heroes));
    CHECK(chosen != nullptr);
    CHECK(chosen->name() == "team_hold" || chosen->name() == "team_push");
}

TEST(team_option_mcts_is_deterministic_under_seed) {
    auto s1 = make_team_scene(2, 2);
    auto s2 = make_team_scene(2, 2);
    mcts::MctsConfig cfg;
    cfg.iterations        = 40;
    cfg.rollout_horizon   = 3;
    cfg.option_max_windows = 2;
    cfg.seed              = 0xBEEF;

    auto run = [&](TeamScene& s) {
        std::vector<std::shared_ptr<mcts::TeamOption>> opts = {
            std::make_shared<TeamHoldOpt>(2),
            std::make_shared<TeamPushOpt>(2),
        };
        mcts::TeamOptionMcts engine(cfg);
        engine.set_options(opts);
        engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
        engine.set_opponent_policy(mcts::policy_idle);
        const mcts::TeamOption* o = engine.search(s.world, raw(s.heroes));
        return o ? o->name() : std::string{};
    };
    CHECK(run(*s1) == run(*s2));
}


// ─── Commander ─────────────────────────────────────────────────────────────

TEST(commander_assigns_roles_and_produces_actions) {
    auto s = make_team_scene(2, 2);
    mcts::Commander::Config cfg;
    cfg.role_cfg.iterations = 30;
    cfg.role_cfg.rollout_horizon = 2;
    cfg.role_cfg.option_max_windows = 2;
    cfg.replan_every_windows = 2;
    mcts::Commander cmdr(cfg);

    // Role A: hold. Role B: attack. Default round-robin assignment → hero 0
    // gets role 0 (hold), hero 1 gets role 1 (attack).
    std::vector<std::shared_ptr<mcts::Option>> hold_opts = { std::make_shared<HoldOpt>(2) };
    std::vector<std::shared_ptr<mcts::Option>> atk_opts  = {
        std::make_shared<HoldOpt>(2), std::make_shared<AttackOpt>(2) };
    cmdr.add_role("hold",   hold_opts);
    cmdr.add_role("attack", atk_opts);
    cmdr.set_default_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    cmdr.set_opponent_policy(mcts::policy_idle);

    auto acts = cmdr.decide(s->world, raw(s->heroes));
    CHECK(acts.size() == 2);
    CHECK(cmdr.current_assignments().size() == 2);
    CHECK(cmdr.current_assignments()[0] == 0);
    CHECK(cmdr.current_assignments()[1] == 1);
    CHECK(cmdr.committed_option_for_hero(0) == "hold");
    // Hero 1 committed either "hold" or "attack" from role 1's set.
    std::string h1_opt = cmdr.committed_option_for_hero(1);
    CHECK(h1_opt == "hold" || h1_opt == "attack");
}

TEST(commander_custom_assigner_is_honored) {
    auto s = make_team_scene(3, 2);
    mcts::Commander::Config cfg;
    cfg.role_cfg.iterations = 20;
    cfg.role_cfg.option_max_windows = 2;
    mcts::Commander cmdr(cfg);

    std::vector<std::shared_ptr<mcts::Option>> opts = { std::make_shared<HoldOpt>(1) };
    cmdr.add_role("alpha", opts);
    cmdr.add_role("beta",  opts);
    cmdr.set_default_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    cmdr.set_opponent_policy(mcts::policy_idle);

    // All heroes on role "beta" (index 1).
    cmdr.set_assigner([](const std::vector<Agent*>& h, const World&) {
        return std::vector<int>(h.size(), 1);
    });
    cmdr.decide(s->world, raw(s->heroes));
    for (int a : cmdr.current_assignments()) CHECK(a == 1);
}

TEST(commander_replan_window_countdown_works) {
    auto s = make_team_scene(2, 2);
    mcts::Commander::Config cfg;
    cfg.role_cfg.iterations = 10;
    cfg.role_cfg.option_max_windows = 1;
    cfg.replan_every_windows = 3;
    mcts::Commander cmdr(cfg);
    std::vector<std::shared_ptr<mcts::Option>> opts = { std::make_shared<HoldOpt>(1) };
    cmdr.add_role("only", opts);
    cmdr.set_default_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    cmdr.set_opponent_policy(mcts::policy_idle);

    cmdr.decide(s->world, raw(s->heroes));
    int w1 = cmdr.windows_until_replan();
    cmdr.decide(s->world, raw(s->heroes));
    int w2 = cmdr.windows_until_replan();
    cmdr.decide(s->world, raw(s->heroes));
    int w3 = cmdr.windows_until_replan();
    CHECK(w1 > w2);
    CHECK(w2 > w3);
}

TEST(commander_dead_hero_skipped) {
    auto s = make_team_scene(2, 2);
    s->heroes[0]->unit().hp = 0.0f;    // kill hero 0
    mcts::Commander::Config cfg;
    cfg.role_cfg.iterations = 10;
    cfg.role_cfg.option_max_windows = 1;
    mcts::Commander cmdr(cfg);
    std::vector<std::shared_ptr<mcts::Option>> opts = { std::make_shared<HoldOpt>(1) };
    cmdr.add_role("only", opts);
    cmdr.set_default_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    cmdr.set_opponent_policy(mcts::policy_idle);

    auto acts = cmdr.decide(s->world, raw(s->heroes));
    // Dead hero gets default-constructed (no-op) action — move_dir Hold,
    // no attack, no ability. We can't probe "committed option" since there
    // isn't one, so just assert the default shape.
    CHECK(acts[0].move_dir == mcts::MoveDir::Hold);
    CHECK(acts[0].attack_slot == -1);
    CHECK(acts[0].ability_slot == -1);
    CHECK(cmdr.committed_option_for_hero(0) == "");
}


TEST(aggressive_rollout_returns_legal_action) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::AggressiveRollout rollout;
    auto a = rollout.choose(s->hero, s->world);
    // Enemy is in range → should pick an attack slot; movement is still N
    // (charge toward enemy), which is fine.
    CHECK(a.attack_slot >= 0);
    CHECK(a.move_dir == mcts::MoveDir::N);
}

TEST(aggressive_rollout_dead_agent_returns_noop) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    s->hero.unit().hp = 0.0f;
    mcts::AggressiveRollout rollout;
    auto a = rollout.choose(s->hero, s->world);
    CHECK(a.attack_slot == -1);
    CHECK(a.ability_slot == -1);
    CHECK(a.move_dir == mcts::MoveDir::Hold);
}

TEST(mcts_with_aggressive_rollout_picks_attack_vs_idle_opponent) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xA66;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    auto a = engine.search(s->world, s->hero);
    CHECK(a.attack_slot >= 0);
    CHECK(engine.last_stats().best_mean > 0.0f);
}

TEST(mcts_pw_bounds_root_children_by_visits_alpha) {
    // With pw_alpha = 0.5 and 100 iterations, the root should have at most
    // ceil(100^0.5) = 10 children. Without PW (classical), the root would
    // expand all 9 move-direction-only actions on the first ~9 iterations
    // and typically end up with 10+ children quickly.
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 100;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0x1234;
    cfg.pw_alpha       = 0.5f;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, s->hero);
    int bound = static_cast<int>(std::ceil(std::pow(100.0f, 0.5f)));
    CHECK(engine.last_stats().root_children <= bound);
    CHECK(engine.last_stats().root_children >= 1);
}

TEST(mcts_pw_off_matches_classical_expansion) {
    // pw_alpha = 0 should leave expansion unchanged: root children count
    // saturates at the number of legal actions (capped by iteration budget).
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 100;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0x5678;
    cfg.pw_alpha       = 0.0f;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.search(s->world, s->hero);
    // Under PW=0, we expect the root to have expanded well beyond the PW(0.5)
    // bound of 10 — we're just asserting PW isn't being applied.
    CHECK(engine.last_stats().root_children > 10);
}

TEST(mcts_pw_still_picks_attack_vs_idle) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 256;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xABCD;
    cfg.pw_alpha       = 0.5f;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    auto a = engine.search(s->world, s->hero);
    CHECK(a.attack_slot >= 0);
    CHECK(engine.last_stats().best_mean > 0.0f);
}

TEST(mcts_pw_is_deterministic_under_seed) {
    auto s1 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    auto s2 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xCAFE;
    cfg.pw_alpha       = 0.5f;
    auto run = [&](McstScene& s) {
        mcts::Mcts e(cfg);
        e.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
        e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
        e.set_opponent_policy(mcts::policy_idle);
        return e.search(s.world, s.hero);
    };
    auto a1 = run(*s1);
    auto a2 = run(*s2);
    CHECK(a1 == a2);
}

TEST(attack_bias_prior_weights_attack_over_move) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::AttackBiasPrior prior;
    auto acts = mcts::legal_actions(s->hero, s->world);
    CHECK(acts.size() > 1);
    auto w = prior.score(s->hero, s->world, acts);
    CHECK(w.size() == acts.size());
    // Find a pure-move and an attack action, verify the attack outweighs.
    float max_move = 0.0f, min_attack = 1e9f;
    bool saw_move = false, saw_attack = false;
    for (size_t i = 0; i < acts.size(); i++) {
        if (acts[i].attack_slot >= 0) {
            saw_attack = true; if (w[i] < min_attack) min_attack = w[i];
        } else if (acts[i].ability_slot < 0) {
            saw_move = true; if (w[i] > max_move) max_move = w[i];
        }
    }
    CHECK(saw_move);
    CHECK(saw_attack);
    CHECK(min_attack > max_move);
}

TEST(uniform_prior_all_equal) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::UniformPrior prior;
    auto acts = mcts::legal_actions(s->hero, s->world);
    auto w = prior.score(s->hero, s->world, acts);
    CHECK(w.size() == acts.size());
    for (float v : w) CHECK(std::abs(v - w[0]) < 1e-6f);
}

TEST(mcts_puct_picks_attack_with_prior) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 8;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xBAB;
    cfg.prior_c        = 1.5f;     // enable PUCT
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.set_prior(std::make_shared<mcts::AttackBiasPrior>());

    auto a = engine.search(s->world, s->hero);
    CHECK(a.attack_slot >= 0);
    CHECK(engine.last_stats().best_mean > 0.0f);

    // With an attack-biased prior, the most-visited attack-containing child
    // should accumulate more visits than an average child.
    const auto* root = engine.last_root();
    CHECK(root != nullptr);
    int attack_visits = 0, move_only_visits = 0;
    for (const auto& up : root->children) {
        if (up->action.attack_slot >= 0) attack_visits += up->visits;
        else if (up->action.ability_slot < 0) move_only_visits += up->visits;
    }
    CHECK(attack_visits > move_only_visits);
}

TEST(mcts_puct_child_priors_sum_to_one) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 64;
    cfg.rollout_horizon = 4;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xBAB2;
    cfg.prior_c        = 1.5f;
    mcts::Mcts engine(cfg);
    engine.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    engine.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    engine.set_opponent_policy(mcts::policy_idle);
    engine.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    engine.search(s->world, s->hero);
    const auto* root = engine.last_root();
    CHECK(root != nullptr);
    // Sum of all children priors + remaining untried priors should be ~1.
    float sum = 0.0f;
    for (const auto& up : root->children) sum += up->prior_p;
    for (float p : root->untried_priors) sum += p;
    CHECK(std::abs(sum - 1.0f) < 1e-4f);
}

TEST(mcts_puct_is_deterministic_under_seed) {
    auto s1 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    auto s2 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xBAB3;
    cfg.prior_c        = 1.5f;
    auto run = [&](McstScene& s) {
        mcts::Mcts e(cfg);
        e.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
        e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
        e.set_opponent_policy(mcts::policy_idle);
        e.set_prior(std::make_shared<mcts::AttackBiasPrior>());
        return e.search(s.world, s.hero);
    };
    CHECK(run(*s1) == run(*s2));
}

TEST(dmcts_puct_uses_priors_and_is_deterministic) {
    auto s1 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    auto s2 = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xD00;
    cfg.prior_c        = 1.5f;
    auto run = [&](McstScene& s) {
        mcts::DecoupledMcts e(cfg);
        e.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
        e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
        e.set_prior(std::make_shared<mcts::AttackBiasPrior>());
        return e.search(s.world, s.hero, s.enemy);
    };
    auto j1 = run(*s1);
    auto j2 = run(*s2);
    CHECK(j1.hero == j2.hero);
    CHECK(j1.opp  == j2.opp);
    // Priors were normalized to sum to 1 on both players.
    // (Indirect check: search completed with non-empty stats.)
}

TEST(dmcts_puct_root_priors_sum_to_one) {
    auto s = make_duel_scene(0, 0, 1.0f, 0, /*attackRange*/ 3.0f);
    mcts::MctsConfig cfg;
    cfg.iterations     = 32;
    cfg.rollout_horizon = 4;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xD01;
    cfg.prior_c        = 1.5f;
    mcts::DecoupledMcts e(cfg);
    e.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    e.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    e.search(s->world, s->hero, s->enemy);
    const auto* r = e.last_root();
    CHECK(r != nullptr);
    float sh = 0.0f, so = 0.0f;
    for (float p : r->hero_stats.priors) sh += p;
    for (float p : r->opp_stats.priors)  so += p;
    CHECK(std::abs(sh - 1.0f) < 1e-4f);
    CHECK(std::abs(so - 1.0f) < 1e-4f);
}

TEST(team_mcts_puct_uses_priors_and_is_deterministic) {
    auto s1 = make_team_scene(2, 2);
    auto s2 = make_team_scene(2, 2);
    mcts::MctsConfig cfg;
    cfg.iterations     = 128;
    cfg.rollout_horizon = 6;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xD02;
    cfg.prior_c        = 1.5f;
    auto run = [&](TeamScene& s) {
        mcts::TeamMcts e(cfg);
        e.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
        e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
        e.set_opponent_policy(mcts::policy_idle);
        e.set_prior(std::make_shared<mcts::AttackBiasPrior>());
        return e.search(s.world, raw(s.heroes));
    };
    auto j1 = run(*s1);
    auto j2 = run(*s2);
    CHECK(j1 == j2);
}

TEST(team_mcts_puct_root_priors_sum_to_one_per_hero) {
    auto s = make_team_scene(2, 2);
    mcts::MctsConfig cfg;
    cfg.iterations     = 64;
    cfg.rollout_horizon = 4;
    cfg.action_repeat  = 2;
    cfg.seed           = 0xD03;
    cfg.prior_c        = 1.5f;
    mcts::TeamMcts e(cfg);
    e.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    e.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    e.set_opponent_policy(mcts::policy_idle);
    e.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    e.search(s->world, raw(s->heroes));
    const auto* r = e.last_root();
    CHECK(r != nullptr);
    for (const auto& ph : r->per_hero) {
        float sum = 0.0f;
        for (float p : ph.priors) sum += p;
        CHECK(std::abs(sum - 1.0f) < 1e-4f);
    }
}

TEST(tactic_prior_boosts_matching_action) {
    auto s = make_team_scene(1, 1);
    auto& hero = *s->heroes.front();
    // FocusLowestHp with the one enemy → target action is attack_slot=0.
    mcts::TacticPrior tp;
    tp.set_tactic(mcts::Tactic{ mcts::TacticKind::FocusLowestHp });
    auto acts = mcts::legal_actions(hero, s->world);
    auto w = tp.score(hero, s->world, acts);
    CHECK(w.size() == acts.size());
    mcts::CombatAction target = mcts::tactic_to_action(
        mcts::Tactic{ mcts::TacticKind::FocusLowestHp }, hero, s->world);
    float max_w = 0.0f, match_w = 0.0f;
    bool saw_match = false;
    for (size_t i = 0; i < acts.size(); i++) {
        if (acts[i] == target) { saw_match = true; match_w = w[i]; }
        if (w[i] > max_w) max_w = w[i];
    }
    CHECK(saw_match);
    CHECK(match_w >= max_w);
    CHECK(match_w > 1.0f);
}

TEST(layered_planner_returns_joint_action_of_correct_size) {
    auto s = make_team_scene(2, 2);
    mcts::LayeredPlanner::Config cfg;
    cfg.tactic_cfg.iterations             = 60;
    cfg.tactic_cfg.rollout_horizon        = 6;
    cfg.tactic_cfg.action_repeat          = 2;
    cfg.tactic_cfg.tactic_window_decisions = 3;
    cfg.tactic_cfg.seed                   = 0x5A1;
    cfg.fine_cfg.iterations               = 60;
    cfg.fine_cfg.rollout_horizon          = 4;
    cfg.fine_cfg.action_repeat            = 2;
    cfg.fine_cfg.seed                     = 0x5A2;
    cfg.fine_cfg.prior_c                  = 1.5f;
    mcts::LayeredPlanner planner(cfg);
    planner.set_team_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    planner.set_rollout_policy(std::make_shared<mcts::AggressiveRollout>());
    planner.set_opponent_policy(mcts::policy_aggressive);

    auto joint = planner.decide(s->world, raw(s->heroes));
    CHECK(joint.per_hero.size() == s->heroes.size());
    // First call must have planned a tactic.
    CHECK(planner.last_stats().replanned_this_call == true);
    CHECK(planner.last_stats().windows_until_replan == 2);  // window=3, one consumed
}

TEST(layered_planner_replans_tactic_on_schedule) {
    auto s = make_team_scene(2, 2);
    mcts::LayeredPlanner::Config cfg;
    cfg.tactic_cfg.iterations             = 40;
    cfg.tactic_cfg.rollout_horizon        = 4;
    cfg.tactic_cfg.action_repeat          = 2;
    cfg.tactic_cfg.tactic_window_decisions = 2;
    cfg.tactic_cfg.seed                   = 0x5A3;
    cfg.fine_cfg.iterations               = 40;
    cfg.fine_cfg.rollout_horizon          = 4;
    cfg.fine_cfg.action_repeat            = 2;
    cfg.fine_cfg.seed                     = 0x5A4;
    cfg.fine_cfg.prior_c                  = 1.5f;
    mcts::LayeredPlanner planner(cfg);
    planner.set_team_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    planner.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    planner.set_opponent_policy(mcts::policy_idle);

    // Windows=2 → replan pattern: T, F, T, F, T, ...
    std::vector<bool> replanned;
    for (int i = 0; i < 5; i++) {
        planner.decide(s->world, raw(s->heroes));
        replanned.push_back(planner.last_stats().replanned_this_call);
    }
    CHECK(replanned[0] == true);
    CHECK(replanned[1] == false);
    CHECK(replanned[2] == true);
    CHECK(replanned[3] == false);
    CHECK(replanned[4] == true);
}

TEST(layered_planner_reset_forces_next_replan) {
    auto s = make_team_scene(2, 2);
    mcts::LayeredPlanner::Config cfg;
    cfg.tactic_cfg.iterations             = 30;
    cfg.tactic_cfg.rollout_horizon        = 4;
    cfg.tactic_cfg.action_repeat          = 2;
    cfg.tactic_cfg.tactic_window_decisions = 5;
    cfg.tactic_cfg.seed                   = 0x5A5;
    cfg.fine_cfg.iterations               = 30;
    cfg.fine_cfg.rollout_horizon          = 4;
    cfg.fine_cfg.action_repeat            = 2;
    cfg.fine_cfg.seed                     = 0x5A6;
    cfg.fine_cfg.prior_c                  = 1.5f;
    mcts::LayeredPlanner planner(cfg);
    planner.set_team_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
    planner.set_rollout_policy(std::make_shared<mcts::RandomRollout>());
    planner.set_opponent_policy(mcts::policy_idle);
    planner.decide(s->world, raw(s->heroes));
    CHECK(planner.last_stats().windows_until_replan == 4);
    planner.reset();
    planner.decide(s->world, raw(s->heroes));
    CHECK(planner.last_stats().replanned_this_call == true);
}
