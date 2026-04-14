// mcts_bench — quantify MCTS strength and speed across config sweeps.
//
// Usage:
//   mcts_bench duel    [--episodes N] [--iterations M] [--budget-ms T]
//                      [--rollout {random|aggressive}] [--opponent {idle|aggressive}]
//                      [--puct C] [--pw A] [--max-ticks K] [--seed S]
//   mcts_bench team    [--episodes N] [--iterations M] [--heroes H] [--enemies E]
//                      [--planner {team|layered}] [--rollout {random|aggressive}]
//                      [--opponent {idle|aggressive}] [--puct C] [--pw A]
//                      [--max-ticks K] [--seed S]
//
// Output is a single TSV row of the form:
//   mode  episodes  wins  losses  draws  mean_hp_delta  mean_search_ms  total_ms
// preceded by a header line. Pipe into awk / csvkit / pandas.

#include "brogameagent/agent.h"
#include "brogameagent/mcts.h"
#include "brogameagent/world.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace brogameagent;

namespace {

struct Args {
    std::string mode = "duel";
    int   episodes    = 20;
    int   iterations  = 200;
    int   budget_ms   = 0;
    int   max_ticks   = 400;       // decision windows per episode
    int   heroes      = 2;
    int   enemies     = 2;
    uint64_t seed     = 0xBEE5CAFEULL;
    std::string rollout  = "random";
    std::string opponent = "aggressive";
    std::string planner  = "team";
    float puct_c = 0.0f;
    float pw_a   = 0.0f;
};

int usage() {
    std::fprintf(stderr,
        "usage: mcts_bench duel|team [flags]\n"
        "  --episodes N       default 20\n"
        "  --iterations M     per-decision iteration cap (default 200)\n"
        "  --budget-ms T      per-decision wall-time cap (default 0)\n"
        "  --max-ticks K      decisions per episode (default 400)\n"
        "  --rollout {random|aggressive}   default random\n"
        "  --opponent {idle|aggressive}    default aggressive\n"
        "  --puct C           enable PUCT with prior_c=C (default 0, off)\n"
        "  --pw A             progressive widening alpha (default 0, off)\n"
        "  --heroes H         team mode only (default 2)\n"
        "  --enemies E        team mode only (default 2)\n"
        "  --planner {team|layered}  team mode only (default team)\n"
        "  --seed S           default 0xBEE5CAFE\n");
    return 2;
}

bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.mode = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string k = argv[i];
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if      (k == "--episodes")   { auto v=need(k.c_str()); if(!v) return false; a.episodes   = std::atoi(v); }
        else if (k == "--iterations") { auto v=need(k.c_str()); if(!v) return false; a.iterations = std::atoi(v); }
        else if (k == "--budget-ms")  { auto v=need(k.c_str()); if(!v) return false; a.budget_ms  = std::atoi(v); }
        else if (k == "--max-ticks")  { auto v=need(k.c_str()); if(!v) return false; a.max_ticks  = std::atoi(v); }
        else if (k == "--rollout")    { auto v=need(k.c_str()); if(!v) return false; a.rollout    = v; }
        else if (k == "--opponent")   { auto v=need(k.c_str()); if(!v) return false; a.opponent   = v; }
        else if (k == "--puct")       { auto v=need(k.c_str()); if(!v) return false; a.puct_c     = static_cast<float>(std::atof(v)); }
        else if (k == "--pw")         { auto v=need(k.c_str()); if(!v) return false; a.pw_a       = static_cast<float>(std::atof(v)); }
        else if (k == "--heroes")     { auto v=need(k.c_str()); if(!v) return false; a.heroes     = std::atoi(v); }
        else if (k == "--enemies")    { auto v=need(k.c_str()); if(!v) return false; a.enemies    = std::atoi(v); }
        else if (k == "--planner")    { auto v=need(k.c_str()); if(!v) return false; a.planner    = v; }
        else if (k == "--seed")       { auto v=need(k.c_str()); if(!v) return false; a.seed       = static_cast<uint64_t>(std::strtoull(v, nullptr, 0)); }
        else { std::fprintf(stderr, "unknown flag: %s\n", k.c_str()); return false; }
    }
    return true;
}

std::shared_ptr<mcts::IRolloutPolicy> make_rollout(const std::string& kind) {
    if (kind == "aggressive") return std::make_shared<mcts::AggressiveRollout>();
    return std::make_shared<mcts::RandomRollout>();
}

mcts::OpponentPolicy make_opponent(const std::string& kind) {
    if (kind == "aggressive") return mcts::policy_aggressive;
    return mcts::policy_idle;
}

// ── Scene builders ────────────────────────────────────────────────────────

struct DuelScene {
    World  world;
    Agent  hero;
    Agent  enemy;
};

std::unique_ptr<DuelScene> build_duel(uint64_t seed) {
    auto s = std::make_unique<DuelScene>();
    s->hero.unit().id = 1;
    s->hero.unit().teamId = 0;
    s->hero.unit().hp = 100.0f;
    s->hero.unit().maxHp = 100.0f;
    s->hero.unit().damage = 10.0f;
    s->hero.unit().attackRange = 3.0f;
    s->hero.unit().attacksPerSec = 2.0f;
    s->hero.setPosition(-1.0f, 0.0f);
    s->hero.setMaxAccel(30.0f);
    s->hero.setMaxTurnRate(10.0f);

    s->enemy.unit().id = 2;
    s->enemy.unit().teamId = 1;
    s->enemy.unit().hp = 80.0f;
    s->enemy.unit().maxHp = 80.0f;
    s->enemy.unit().damage = 8.0f;
    s->enemy.unit().attackRange = 3.0f;
    s->enemy.unit().attacksPerSec = 1.5f;
    s->enemy.setPosition(1.0f, 0.0f);
    s->enemy.setMaxAccel(30.0f);
    s->enemy.setMaxTurnRate(10.0f);

    s->world.addAgent(&s->hero);
    s->world.addAgent(&s->enemy);
    s->world.seed(seed);
    return s;
}

struct TeamScene {
    World world;
    std::vector<std::unique_ptr<Agent>> heroes;
    std::vector<std::unique_ptr<Agent>> enemies;
};

std::unique_ptr<TeamScene> build_team(int nh, int ne, uint64_t seed) {
    auto s = std::make_unique<TeamScene>();
    int next = 1;
    for (int i = 0; i < nh; i++) {
        auto a = std::make_unique<Agent>();
        a->unit().id = next++;
        a->unit().teamId = 0;
        a->unit().hp = 100.0f;
        a->unit().maxHp = 100.0f;
        a->unit().damage = 10.0f;
        a->unit().attackRange = 3.0f;
        a->unit().attacksPerSec = 2.0f;
        a->setPosition(-1.5f + 0.4f * i, 0.3f * i);
        a->setMaxAccel(30.0f);
        a->setMaxTurnRate(10.0f);
        s->world.addAgent(a.get());
        s->heroes.push_back(std::move(a));
    }
    for (int i = 0; i < ne; i++) {
        auto a = std::make_unique<Agent>();
        a->unit().id = next++;
        a->unit().teamId = 1;
        a->unit().hp = 60.0f;
        a->unit().maxHp = 60.0f;
        a->unit().damage = 6.0f;
        a->unit().attackRange = 3.0f;
        a->unit().attacksPerSec = 1.0f;
        a->setPosition(1.5f + 0.4f * i, 0.3f * i);
        a->setMaxAccel(30.0f);
        a->setMaxTurnRate(10.0f);
        s->world.addAgent(a.get());
        s->enemies.push_back(std::move(a));
    }
    s->world.seed(seed);
    return s;
}

// ── Per-episode drivers ───────────────────────────────────────────────────

struct EpisodeResult {
    int   outcome      = 0;     // +1 win, -1 loss, 0 draw
    float hp_delta     = 0.0f;  // hero-side HP fraction minus enemy-side
    long  total_search_us = 0;
    int   decisions    = 0;
};

float team_hp_frac(const std::vector<Agent*>& agents, int team_id) {
    float sum = 0.0f;
    int   count = 0;
    for (Agent* a : agents) {
        if (a->unit().teamId != team_id) continue;
        count++;
        if (a->unit().alive())
            sum += a->unit().hp / std::max(1e-6f, a->unit().maxHp);
    }
    return count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

EpisodeResult run_duel_episode(const Args& a, int episode_idx) {
    EpisodeResult r;
    uint64_t ep_seed = a.seed + static_cast<uint64_t>(episode_idx);
    auto s = build_duel(ep_seed);

    mcts::MctsConfig cfg;
    cfg.iterations     = a.iterations;
    cfg.budget_ms      = a.budget_ms;
    cfg.rollout_horizon = 16;
    cfg.action_repeat  = 4;
    cfg.seed           = ep_seed;
    cfg.prior_c        = a.puct_c;
    cfg.pw_alpha       = a.pw_a;

    mcts::Mcts hero_mcts(cfg);
    hero_mcts.set_evaluator(std::make_shared<mcts::HpDeltaEvaluator>());
    hero_mcts.set_rollout_policy(make_rollout(a.rollout));
    hero_mcts.set_opponent_policy(make_opponent(a.opponent));
    if (a.puct_c > 0.0f) hero_mcts.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    auto opp_policy = make_opponent(a.opponent);

    const float dt = cfg.sim_dt;
    for (int t = 0; t < a.max_ticks; t++) {
        if (!s->hero.unit().alive() || !s->enemy.unit().alive()) break;

        auto t0 = std::chrono::steady_clock::now();
        mcts::CombatAction h_act = hero_mcts.search(s->world, s->hero);
        auto t1 = std::chrono::steady_clock::now();
        r.total_search_us += std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
        r.decisions++;

        mcts::CombatAction o_act = s->enemy.unit().alive()
            ? opp_policy(s->enemy, s->world)
            : mcts::CombatAction{};

        for (int w = 0; w < cfg.action_repeat; w++) {
            mcts::apply(s->hero,  s->world, h_act, dt);
            mcts::apply(s->enemy, s->world, o_act, dt);
            s->world.stepProjectiles(dt);
            s->world.cullProjectiles();
            if (!s->hero.unit().alive() || !s->enemy.unit().alive()) break;
        }
    }
    bool hero_alive  = s->hero.unit().alive();
    bool enemy_alive = s->enemy.unit().alive();
    if (hero_alive && !enemy_alive)     r.outcome = +1;
    else if (!hero_alive && enemy_alive) r.outcome = -1;
    else                                 r.outcome =  0;

    float hero_frac  = hero_alive  ? s->hero.unit().hp  / s->hero.unit().maxHp  : 0.0f;
    float enemy_frac = enemy_alive ? s->enemy.unit().hp / s->enemy.unit().maxHp : 0.0f;
    r.hp_delta = hero_frac - enemy_frac;
    return r;
}

EpisodeResult run_team_episode(const Args& a, int episode_idx) {
    EpisodeResult r;
    uint64_t ep_seed = a.seed + static_cast<uint64_t>(episode_idx);
    auto s = build_team(a.heroes, a.enemies, ep_seed);

    mcts::MctsConfig fine_cfg;
    fine_cfg.iterations = a.iterations;
    fine_cfg.budget_ms  = a.budget_ms;
    fine_cfg.rollout_horizon = 12;
    fine_cfg.action_repeat   = 4;
    fine_cfg.seed       = ep_seed;
    fine_cfg.prior_c    = a.puct_c;
    fine_cfg.pw_alpha   = a.pw_a;

    std::vector<Agent*> heroes;
    for (auto& h : s->heroes) heroes.push_back(h.get());
    auto opp_policy = make_opponent(a.opponent);

    // Planner setup
    mcts::TeamMcts   team_engine;
    mcts::LayeredPlanner layered;
    if (a.planner == "layered") {
        mcts::LayeredPlanner::Config lcfg;
        lcfg.tactic_cfg.iterations = std::max(20, a.iterations / 4);
        lcfg.tactic_cfg.rollout_horizon = 8;
        lcfg.tactic_cfg.action_repeat = 4;
        lcfg.tactic_cfg.tactic_window_decisions = 4;
        lcfg.tactic_cfg.seed = ep_seed ^ 0xABCDULL;
        lcfg.fine_cfg = fine_cfg;
        if (a.puct_c <= 0.0f) lcfg.fine_cfg.prior_c = 1.5f;  // bias requires PUCT
        layered.set_config(lcfg);
        layered.set_team_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
        layered.set_rollout_policy(make_rollout(a.rollout));
        layered.set_opponent_policy(opp_policy);
    } else {
        team_engine.set_config(fine_cfg);
        team_engine.set_evaluator(std::make_shared<mcts::TeamHpDeltaEvaluator>());
        team_engine.set_rollout_policy(make_rollout(a.rollout));
        team_engine.set_opponent_policy(opp_policy);
        if (a.puct_c > 0.0f)
            team_engine.set_prior(std::make_shared<mcts::AttackBiasPrior>());
    }

    const float dt = fine_cfg.sim_dt;
    auto any_alive = [&](int team_id) {
        for (Agent* a2 : s->world.agents())
            if (a2->unit().alive() && a2->unit().teamId == team_id) return true;
        return false;
    };

    for (int t = 0; t < a.max_ticks; t++) {
        if (!any_alive(0) || !any_alive(1)) break;

        auto t0 = std::chrono::steady_clock::now();
        mcts::TeamMcts::JointAction joint;
        if (a.planner == "layered") {
            joint = layered.decide(s->world, heroes);
        } else {
            team_engine.reset_tree();
            joint = team_engine.search(s->world, heroes);
        }
        auto t1 = std::chrono::steady_clock::now();
        r.total_search_us += std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
        r.decisions++;

        // Opponent actions each decision window.
        std::vector<std::pair<Agent*, mcts::CombatAction>> opp_acts;
        for (Agent* ag : s->world.agents()) {
            if (ag->unit().teamId == 0) continue;
            mcts::CombatAction oa{};
            if (ag->unit().alive()) oa = opp_policy(*ag, s->world);
            opp_acts.emplace_back(ag, oa);
        }

        for (int w = 0; w < fine_cfg.action_repeat; w++) {
            for (size_t i = 0; i < heroes.size(); i++) {
                mcts::apply(*heroes[i], s->world,
                            i < joint.per_hero.size() ? joint.per_hero[i] : mcts::CombatAction{},
                            dt);
            }
            for (auto& [ag, oa] : opp_acts) mcts::apply(*ag, s->world, oa, dt);
            s->world.stepProjectiles(dt);
            s->world.cullProjectiles();
            if (!any_alive(0) || !any_alive(1)) break;
        }
    }

    std::vector<Agent*> all;
    for (Agent* a2 : s->world.agents()) all.push_back(a2);
    float hero_frac  = team_hp_frac(all, 0);
    float enemy_frac = team_hp_frac(all, 1);
    r.hp_delta = hero_frac - enemy_frac;
    bool hero_alive  = any_alive(0);
    bool enemy_alive = any_alive(1);
    if (hero_alive && !enemy_alive)      r.outcome = +1;
    else if (!hero_alive && enemy_alive) r.outcome = -1;
    else                                 r.outcome =  0;
    return r;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return usage();

    int wins = 0, losses = 0, draws = 0;
    double sum_hp = 0.0;
    long   total_search_us = 0;
    long   total_decisions = 0;

    auto t_start = std::chrono::steady_clock::now();
    for (int e = 0; e < a.episodes; e++) {
        EpisodeResult r = (a.mode == "duel")
            ? run_duel_episode(a, e)
            : run_team_episode(a, e);
        if (r.outcome > 0)      wins++;
        else if (r.outcome < 0) losses++;
        else                    draws++;
        sum_hp          += r.hp_delta;
        total_search_us += r.total_search_us;
        total_decisions += r.decisions;
    }
    auto t_end = std::chrono::steady_clock::now();
    long total_ms = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());

    double mean_hp     = a.episodes > 0 ? sum_hp / a.episodes : 0.0;
    double mean_search_ms = total_decisions > 0
        ? (static_cast<double>(total_search_us) / 1000.0) / total_decisions
        : 0.0;

    std::printf("mode\tepisodes\titerations\twins\tlosses\tdraws\tmean_hp_delta\tmean_search_ms\ttotal_ms\n");
    std::printf("%s\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%ld\n",
        a.mode.c_str(), a.episodes, a.iterations,
        wins, losses, draws, mean_hp, mean_search_ms, total_ms);
    return 0;
}
