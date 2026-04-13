// Python bindings for brogameagent.
//
// Build via CMake with -DBROGAMEAGENT_PYTHON=ON. Produces a `brogameagent`
// extension module. Observation / action-mask builders return numpy arrays.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <brogameagent/brogameagent.h>

namespace py = pybind11;
using namespace brogameagent;

PYBIND11_MODULE(brogameagent, m) {
    m.doc() = "brogameagent: 2D MOBA-style agent library with RL substrate";

    // --- Primitive types ---

    py::class_<Vec2>(m, "Vec2")
        .def(py::init<>())
        .def(py::init<float, float>(), py::arg("x"), py::arg("z"))
        .def_readwrite("x", &Vec2::x)
        .def_readwrite("z", &Vec2::z)
        .def("length", &Vec2::length)
        .def("__repr__", [](const Vec2& v) {
            return "Vec2(" + std::to_string(v.x) + ", " + std::to_string(v.z) + ")";
        });

    py::class_<AABB>(m, "AABB")
        .def(py::init<>())
        .def(py::init([](float cx, float cz, float hw, float hd) {
            return AABB{cx, cz, hw, hd};
        }), py::arg("cx"), py::arg("cz"), py::arg("hw"), py::arg("hd"))
        .def_readwrite("cx", &AABB::cx)
        .def_readwrite("cz", &AABB::cz)
        .def_readwrite("hw", &AABB::hw)
        .def_readwrite("hd", &AABB::hd);

    py::class_<AimResult>(m, "AimResult")
        .def_readwrite("yaw", &AimResult::yaw)
        .def_readwrite("pitch", &AimResult::pitch);

    // --- Combat ---

    py::enum_<DamageKind>(m, "DamageKind")
        .value("Physical", DamageKind::Physical)
        .value("Magical", DamageKind::Magical)
        .value("True_", DamageKind::True);

    py::class_<DamageEvent>(m, "DamageEvent")
        .def_readonly("attacker_id", &DamageEvent::attackerId)
        .def_readonly("target_id", &DamageEvent::targetId)
        .def_readonly("amount", &DamageEvent::amount)
        .def_readonly("kind", &DamageEvent::kind)
        .def_readonly("killed", &DamageEvent::killed);

    py::class_<Unit>(m, "Unit")
        .def(py::init<>())
        .def_readwrite("id", &Unit::id)
        .def_readwrite("team_id", &Unit::teamId)
        .def_readwrite("hp", &Unit::hp)
        .def_readwrite("max_hp", &Unit::maxHp)
        .def_readwrite("mana", &Unit::mana)
        .def_readwrite("max_mana", &Unit::maxMana)
        .def_readwrite("damage", &Unit::damage)
        .def_readwrite("attack_range", &Unit::attackRange)
        .def_readwrite("attacks_per_sec", &Unit::attacksPerSec)
        .def_readwrite("armor", &Unit::armor)
        .def_readwrite("magic_resist", &Unit::magicResist)
        .def_readwrite("move_speed", &Unit::moveSpeed)
        .def_readwrite("radius", &Unit::radius)
        .def_readwrite("attack_kind", &Unit::attackKind)
        .def_readwrite("attack_cooldown", &Unit::attackCooldown)
        .def_readwrite("mana_regen_per_sec", &Unit::manaRegenPerSec)
        .def_readwrite("armor_bonus", &Unit::armorBonus)
        .def_readwrite("armor_bonus_remaining", &Unit::armorBonusRemaining)
        .def_readwrite("magic_resist_bonus", &Unit::magicResistBonus)
        .def_readwrite("magic_resist_bonus_remaining", &Unit::magicResistBonusRemaining)
        .def_readwrite("damage_mul", &Unit::damageMul)
        .def_readwrite("damage_mul_remaining", &Unit::damageMulRemaining)
        .def_readwrite("attacks_mul", &Unit::attacksMul)
        .def_readwrite("attacks_mul_remaining", &Unit::attacksMulRemaining)
        .def_readwrite("move_speed_mul", &Unit::moveSpeedMul)
        .def_readwrite("move_speed_mul_remaining", &Unit::moveSpeedMulRemaining)
        .def_readwrite("stealth_chance", &Unit::stealthChance)
        .def_readwrite("stealth_chance_remaining", &Unit::stealthChanceRemaining)
        .def_readwrite("dot_dps", &Unit::dotDps)
        .def_readwrite("dot_remaining", &Unit::dotRemaining)
        .def_readwrite("hot_rate", &Unit::hotRate)
        .def_readwrite("hot_remaining", &Unit::hotRemaining)
        .def_property_readonly("alive", &Unit::alive)
        .def("take_damage", &Unit::takeDamage,
             py::arg("amount"), py::arg("kind"))
        .def("tick_cooldowns", &Unit::tickCooldowns, py::arg("dt"))
        .def("set_ability_slot", [](Unit& u, int slot, int abilityId) {
            if (slot < 0 || slot >= Unit::MAX_ABILITIES)
                throw py::index_error("ability slot out of range");
            u.abilitySlot[slot] = abilityId;
        })
        .def("get_ability_slot", [](const Unit& u, int slot) {
            if (slot < 0 || slot >= Unit::MAX_ABILITIES)
                throw py::index_error("ability slot out of range");
            return u.abilitySlot[slot];
        })
        .def("set_ability_cooldown", [](Unit& u, int slot, float cd) {
            if (slot < 0 || slot >= Unit::MAX_ABILITIES)
                throw py::index_error("ability slot out of range");
            u.abilityCooldowns[slot] = cd;
        })
        .def("get_ability_cooldown", [](const Unit& u, int slot) {
            if (slot < 0 || slot >= Unit::MAX_ABILITIES)
                throw py::index_error("ability slot out of range");
            return u.abilityCooldowns[slot];
        })
        .def_readonly_static("MAX_ABILITIES", &Unit::MAX_ABILITIES);

    // --- Projectiles ---

    py::enum_<ProjectileMode>(m, "ProjectileMode")
        .value("Single", ProjectileMode::Single)
        .value("Pierce", ProjectileMode::Pierce)
        .value("AoE", ProjectileMode::AoE);

    py::class_<Projectile>(m, "Projectile")
        .def(py::init<>())
        .def_readwrite("id", &Projectile::id)
        .def_readwrite("owner_id", &Projectile::ownerId)
        .def_readwrite("team_id", &Projectile::teamId)
        .def_readwrite("target_id", &Projectile::targetId)
        .def_readwrite("x", &Projectile::x)
        .def_readwrite("z", &Projectile::z)
        .def_readwrite("vx", &Projectile::vx)
        .def_readwrite("vz", &Projectile::vz)
        .def_readwrite("speed", &Projectile::speed)
        .def_readwrite("radius", &Projectile::radius)
        .def_readwrite("damage", &Projectile::damage)
        .def_readwrite("kind", &Projectile::kind)
        .def_readwrite("remaining_life", &Projectile::remainingLife)
        .def_readwrite("mode", &Projectile::mode)
        .def_readwrite("splash_radius", &Projectile::splashRadius)
        .def_readwrite("max_hits", &Projectile::maxHits)
        .def_readwrite("alive", &Projectile::alive);

    // --- AgentAction ---

    py::class_<AgentAction>(m, "AgentAction")
        .def(py::init<>())
        .def(py::init([](float move_x, float move_z, float aim_yaw, float aim_pitch,
                         int attack_target_id, int use_ability_id) {
            AgentAction a;
            a.moveX = move_x; a.moveZ = move_z;
            a.aimYaw = aim_yaw; a.aimPitch = aim_pitch;
            a.attackTargetId = attack_target_id;
            a.useAbilityId = use_ability_id;
            return a;
        }),
            py::arg("move_x") = 0.0f, py::arg("move_z") = 0.0f,
            py::arg("aim_yaw") = 0.0f, py::arg("aim_pitch") = 0.0f,
            py::arg("attack_target_id") = -1, py::arg("use_ability_id") = -1)
        .def_readwrite("move_x", &AgentAction::moveX)
        .def_readwrite("move_z", &AgentAction::moveZ)
        .def_readwrite("aim_yaw", &AgentAction::aimYaw)
        .def_readwrite("aim_pitch", &AgentAction::aimPitch)
        .def_readwrite("attack_target_id", &AgentAction::attackTargetId)
        .def_readwrite("use_ability_id", &AgentAction::useAbilityId);

    // --- NavGrid ---

    py::class_<NavGrid>(m, "NavGrid")
        .def(py::init<float, float, float, float, float>(),
             py::arg("min_x"), py::arg("min_z"),
             py::arg("max_x"), py::arg("max_z"), py::arg("cell_size"))
        .def("add_obstacle", &NavGrid::addObstacle,
             py::arg("box"), py::arg("padding") = 0.0f)
        .def("is_walkable", &NavGrid::isWalkable, py::arg("x"), py::arg("z"))
        .def("find_path", &NavGrid::findPath, py::arg("from_"), py::arg("to"))
        .def("has_grid_los", &NavGrid::hasGridLOS)
        .def_property_readonly("width", &NavGrid::width)
        .def_property_readonly("height", &NavGrid::height)
        .def_property_readonly("cell_size", &NavGrid::cellSize);

    // --- Agent ---

    py::class_<Agent>(m, "Agent")
        .def(py::init<>())
        .def("set_nav_grid", &Agent::setNavGrid, py::keep_alive<1, 2>())
        .def("set_position", &Agent::setPosition, py::arg("x"), py::arg("z"))
        .def("set_max_accel", &Agent::setMaxAccel)
        .def("set_max_turn_rate", &Agent::setMaxTurnRate)
        .def("set_speed", &Agent::setSpeed)
        .def("set_radius", &Agent::setRadius)
        .def_property_readonly("unit",
            static_cast<Unit& (Agent::*)()>(&Agent::unit),
            py::return_value_policy::reference_internal)
        .def("set_target", &Agent::setTarget, py::arg("x"), py::arg("z"))
        .def("clear_target", &Agent::clearTarget)
        .def("update", &Agent::update, py::arg("dt"))
        .def("apply_action", &Agent::applyAction, py::arg("action"), py::arg("dt"))
        .def_property_readonly("x", &Agent::x)
        .def_property_readonly("z", &Agent::z)
        .def_property_readonly("yaw", &Agent::yaw)
        .def_property_readonly("aim_yaw", &Agent::aimYaw)
        .def_property_readonly("aim_pitch", &Agent::aimPitch)
        .def("aim_at", &Agent::aimAt,
             py::arg("tx"), py::arg("ty"), py::arg("tz"), py::arg("eye_height"))
        .def_property_readonly("has_target", &Agent::hasTarget)
        .def_property_readonly("at_target", &Agent::atTarget)
        .def_property_readonly("path", &Agent::path)
        .def_property_readonly("current_waypoint", &Agent::currentWaypoint)
        .def_property_readonly("velocity", &Agent::velocity);

    // --- AbilitySpec ---

    py::class_<AbilitySpec>(m, "AbilitySpec")
        .def(py::init<>())
        .def_readwrite("cooldown", &AbilitySpec::cooldown)
        .def_readwrite("mana_cost", &AbilitySpec::manaCost)
        .def_readwrite("range", &AbilitySpec::range)
        .def_readwrite("fn", &AbilitySpec::fn);

    // --- World ---

    py::class_<World>(m, "World")
        .def(py::init<>())
        .def("add_agent", &World::addAgent, py::keep_alive<1, 2>())
        .def("remove_agent", &World::removeAgent)
        .def("add_obstacle", &World::addObstacle)
        .def("tick", &World::tick, py::arg("dt"))
        .def("step_projectiles", &World::stepProjectiles, py::arg("dt"))
        .def("cull_projectiles", &World::cullProjectiles)
        .def_property_readonly("agents", &World::agents,
            py::return_value_policy::reference_internal)
        .def_property_readonly("obstacles", &World::obstacles,
            py::return_value_policy::reference_internal)
        .def_property_readonly("projectiles", &World::projectiles,
            py::return_value_policy::reference_internal)
        .def("enemies_of", &World::enemiesOf,
            py::return_value_policy::reference_internal)
        .def("allies_of", &World::alliesOf,
            py::return_value_policy::reference_internal)
        .def("nearest_enemy", &World::nearestEnemy,
            py::return_value_policy::reference_internal)
        .def("enemies_in_range", &World::enemiesInRange,
            py::return_value_policy::reference_internal)
        .def("find_by_id", &World::findById,
            py::return_value_policy::reference_internal)
        // Abilities / combat
        .def("register_ability", &World::registerAbility)
        .def("has_ability", &World::hasAbility)
        .def("resolve_attack", &World::resolveAttack)
        .def("resolve_ability", &World::resolveAbility,
             py::arg("caster"), py::arg("slot"), py::arg("target_id"))
        .def("apply_action", &World::applyAction,
             py::arg("agent"), py::arg("action"), py::arg("dt"))
        // Projectiles
        .def("spawn_projectile", &World::spawnProjectile)
        // Damage log
        .def("deal_damage", &World::dealDamage)
        .def("deal_env_damage", &World::dealEnvDamage)
        .def_property_readonly("events", &World::events,
            py::return_value_policy::reference_internal)
        .def("clear_events", &World::clearEvents)
        // RNG
        .def("seed", &World::seed)
        .def("rand_float_01", &World::randFloat01)
        .def("rand_range", &World::randRange)
        .def("rand_int", &World::randInt)
        .def("chance", &World::chance)
        // Snapshot
        .def("snapshot", &World::snapshot)
        .def("restore", &World::restore);

    // --- Snapshot types (mostly opaque, but fields accessible for debugging) ---

    py::class_<AgentSnapshot>(m, "AgentSnapshot")
        .def_readonly("id", &AgentSnapshot::id)
        .def_readonly("x", &AgentSnapshot::x)
        .def_readonly("z", &AgentSnapshot::z)
        .def_readonly("unit", &AgentSnapshot::unit);

    py::class_<WorldSnapshot>(m, "WorldSnapshot")
        .def_readonly("agents", &WorldSnapshot::agents)
        .def_readonly("projectiles", &WorldSnapshot::projectiles)
        .def_readonly("next_projectile_id", &WorldSnapshot::nextProjectileId);

    // --- RewardTracker ---

    py::class_<RewardTracker::Delta>(m, "RewardDelta")
        .def_readonly("damage_dealt", &RewardTracker::Delta::damageDealt)
        .def_readonly("damage_taken", &RewardTracker::Delta::damageTaken)
        .def_readonly("kills", &RewardTracker::Delta::kills)
        .def_readonly("deaths", &RewardTracker::Delta::deaths)
        .def_readonly("distance_travelled", &RewardTracker::Delta::distanceTravelled);

    py::class_<RewardTracker>(m, "RewardTracker")
        .def(py::init<>())
        .def("reset", &RewardTracker::reset)
        .def("consume", &RewardTracker::consume);

    // --- Simulation ---

    py::class_<Simulation>(m, "Simulation")
        .def(py::init<World&>(), py::keep_alive<1, 2>())
        .def("add_policy", &Simulation::addPolicy)
        .def("remove_policy", &Simulation::removePolicy)
        .def("step", &Simulation::step, py::arg("dt"))
        .def("run_steps", &Simulation::runSteps, py::arg("dt"), py::arg("n"))
        .def_property_readonly("steps", &Simulation::steps)
        .def_property_readonly("elapsed", &Simulation::elapsed)
        .def("reset_counters", &Simulation::resetCounters);

    // --- Observation (submodule) ---

    auto obs = m.def_submodule("observation", "Ego-centric NN observation builder");
    obs.attr("K_ENEMIES")      = observation::K_ENEMIES;
    obs.attr("K_ALLIES")       = observation::K_ALLIES;
    obs.attr("SELF_FEATURES")  = observation::SELF_FEATURES;
    obs.attr("ENEMY_FEATURES") = observation::ENEMY_FEATURES;
    obs.attr("ALLY_FEATURES")  = observation::ALLY_FEATURES;
    obs.attr("OBS_RANGE")      = observation::OBS_RANGE;
    obs.attr("TOTAL")          = observation::TOTAL;

    obs.def("build", [](const Agent& self, const World& world) {
        py::array_t<float> out(observation::TOTAL);
        observation::build(self, world, out.mutable_data());
        return out;
    }, py::arg("agent"), py::arg("world"),
       "Return a (TOTAL,) float32 numpy observation for `agent` in `world`.");

    obs.def("build_into", [](const Agent& self, const World& world,
                             py::array_t<float, py::array::c_style | py::array::forcecast> out) {
        if (out.size() < observation::TOTAL)
            throw py::value_error("output buffer too small");
        observation::build(self, world, out.mutable_data());
    }, py::arg("agent"), py::arg("world"), py::arg("out"),
       "Write the observation into an existing numpy buffer.");

    // --- Action mask (submodule) ---

    auto am = m.def_submodule("action_mask", "Policy action-mask builder");
    am.attr("N_ENEMY_SLOTS")   = action_mask::N_ENEMY_SLOTS;
    am.attr("N_ABILITY_SLOTS") = action_mask::N_ABILITY_SLOTS;
    am.attr("TOTAL")           = action_mask::TOTAL;

    am.def("build", [](const Agent& self, const World& world) {
        py::array_t<float> mask(action_mask::TOTAL);
        py::array_t<int>   ids(action_mask::N_ENEMY_SLOTS);
        action_mask::build(self, world, mask.mutable_data(), ids.mutable_data());
        return py::make_tuple(mask, ids);
    }, py::arg("agent"), py::arg("world"),
       "Return (mask[TOTAL], enemy_ids[N_ENEMY_SLOTS]) numpy arrays.");

    // --- Free functions ---

    m.def("wrap_angle", &wrapAngle);
    m.def("angle_delta", &angleDelta);

    // Steering
    m.def("seek", &seek);
    m.def("arrive", &arrive);
    m.def("flee", &flee);
    m.def("pursue", &pursue);
    m.def("evade", &evade);

    // Perception
    m.def("has_line_of_sight", [](Vec2 from, Vec2 to, std::vector<AABB> obstacles) {
        return hasLineOfSight(from, to, obstacles.data(),
                              static_cast<int>(obstacles.size()));
    });
    m.def("can_see", [](Vec2 from, Vec2 to,
                        float facing_yaw, float fov, float max_range,
                        std::vector<AABB> obstacles) {
        return canSee(from, to, facing_yaw, fov, max_range,
                      obstacles.data(), static_cast<int>(obstacles.size()));
    });
    m.def("compute_aim", &computeAim);
    m.def("compute_lead_aim", &computeLeadAim);

    py::class_<LeadAimResult>(m, "LeadAimResult")
        .def_readonly("aim", &LeadAimResult::aim)
        .def_readonly("valid", &LeadAimResult::valid)
        .def_readonly("time_to_hit", &LeadAimResult::timeToHit);

    // --- Recorder ---

    py::class_<Recorder>(m, "Recorder")
        .def(py::init<>())
        .def("open", &Recorder::open,
             py::arg("path"), py::arg("episode_id"), py::arg("seed"), py::arg("dt"))
        .def("is_open", &Recorder::isOpen)
        .def("write_roster", &Recorder::writeRoster)
        .def("record_frame", &Recorder::recordFrame,
             py::arg("step_idx"), py::arg("elapsed"), py::arg("world"))
        .def("close", &Recorder::close)
        .def_property_readonly("frame_count", &Recorder::frameCount);

    // --- ReplayReader (for analysis in Python) ---

    py::class_<replay::FileHeader>(m, "ReplayFileHeader")
        .def_readonly("magic",      &replay::FileHeader::magic)
        .def_readonly("version",    &replay::FileHeader::version)
        .def_readonly("episode_id", &replay::FileHeader::episodeId)
        .def_readonly("seed",       &replay::FileHeader::seed)
        .def_readonly("dt",         &replay::FileHeader::dt);

    py::class_<replay::AgentStatic>(m, "ReplayAgentStatic")
        .def_readonly("id",           &replay::AgentStatic::id)
        .def_readonly("team_id",      &replay::AgentStatic::teamId)
        .def_readonly("max_hp",       &replay::AgentStatic::maxHp)
        .def_readonly("max_mana",     &replay::AgentStatic::maxMana)
        .def_readonly("radius",       &replay::AgentStatic::radius)
        .def_readonly("attack_range", &replay::AgentStatic::attackRange);

    py::class_<replay::FrameHeader>(m, "ReplayFrameHeader")
        .def_readonly("step_idx",    &replay::FrameHeader::stepIdx)
        .def_readonly("elapsed",     &replay::FrameHeader::elapsed)
        .def_readonly("live_count",  &replay::FrameHeader::liveCount)
        .def_readonly("proj_count",  &replay::FrameHeader::projCount)
        .def_readonly("event_count", &replay::FrameHeader::eventCount);

    py::class_<replay::AgentState>(m, "ReplayAgentState")
        .def_readonly("id",     &replay::AgentState::id)
        .def_readonly("x",      &replay::AgentState::x)
        .def_readonly("z",      &replay::AgentState::z)
        .def_readonly("vx",     &replay::AgentState::vx)
        .def_readonly("vz",     &replay::AgentState::vz)
        .def_readonly("yaw",    &replay::AgentState::yaw)
        .def_readonly("aim_yaw",&replay::AgentState::aimYaw)
        .def_readonly("hp",     &replay::AgentState::hp)
        .def_readonly("mana",   &replay::AgentState::mana)
        .def_readonly("attack_cooldown", &replay::AgentState::attackCooldown)
        .def_property_readonly("alive",
            [](const replay::AgentState& s) {
                return (s.flags & replay::AGENT_FLAG_ALIVE) != 0;
            });

    py::class_<replay::ProjectileState>(m, "ReplayProjectileState")
        .def_readonly("id",       &replay::ProjectileState::id)
        .def_readonly("owner_id", &replay::ProjectileState::ownerId)
        .def_readonly("team_id",  &replay::ProjectileState::teamId)
        .def_readonly("x",        &replay::ProjectileState::x)
        .def_readonly("z",        &replay::ProjectileState::z)
        .def_readonly("vx",       &replay::ProjectileState::vx)
        .def_readonly("vz",       &replay::ProjectileState::vz)
        .def_readonly("mode",     &replay::ProjectileState::mode)
        .def_readonly("alive",    &replay::ProjectileState::alive);

    py::class_<replay::DamageEventRec>(m, "ReplayDamageEvent")
        .def_readonly("attacker_id", &replay::DamageEventRec::attackerId)
        .def_readonly("target_id",   &replay::DamageEventRec::targetId)
        .def_readonly("amount",      &replay::DamageEventRec::amount)
        .def_readonly("kind",        &replay::DamageEventRec::kind)
        .def_readonly("killed",      &replay::DamageEventRec::killed);

    py::class_<ReplayReader::Frame>(m, "ReplayFrame")
        .def_readonly("header",      &ReplayReader::Frame::header)
        .def_readonly("agents",      &ReplayReader::Frame::agents)
        .def_readonly("projectiles", &ReplayReader::Frame::projectiles)
        .def_readonly("events",      &ReplayReader::Frame::events);

    py::class_<ReplayReader::TrajectoryPoint>(m, "ReplayTrajectoryPoint")
        .def_readonly("step_idx", &ReplayReader::TrajectoryPoint::stepIdx)
        .def_readonly("elapsed",  &ReplayReader::TrajectoryPoint::elapsed)
        .def_readonly("x",        &ReplayReader::TrajectoryPoint::x)
        .def_readonly("z",        &ReplayReader::TrajectoryPoint::z)
        .def_readonly("hp",       &ReplayReader::TrajectoryPoint::hp)
        .def_readonly("alive",    &ReplayReader::TrajectoryPoint::alive);

    py::class_<ReplayReader::DamageSummary>(m, "ReplayDamageSummary")
        .def_readonly("attacker_id",  &ReplayReader::DamageSummary::attackerId)
        .def_readonly("target_id",    &ReplayReader::DamageSummary::targetId)
        .def_readonly("total_damage", &ReplayReader::DamageSummary::totalDamage)
        .def_readonly("hits",         &ReplayReader::DamageSummary::hits)
        .def_readonly("kills",        &ReplayReader::DamageSummary::kills);

    // --- VecSimulation ---

    py::class_<VecSimulation::Config>(m, "VecSimConfig")
        .def(py::init<>())
        .def_readwrite("num_envs",              &VecSimulation::Config::numEnvs)
        .def_readwrite("arena_half_size",       &VecSimulation::Config::arenaHalfSize)
        .def_readwrite("min_spawn_dist",        &VecSimulation::Config::minSpawnDist)
        .def_readwrite("max_spawn_dist",        &VecSimulation::Config::maxSpawnDist)
        .def_readwrite("dt",                    &VecSimulation::Config::dt)
        .def_readwrite("max_steps_per_episode", &VecSimulation::Config::maxStepsPerEpisode)
        .def_readwrite("hp",                    &VecSimulation::Config::hp)
        .def_readwrite("max_mana",              &VecSimulation::Config::maxMana)
        .def_readwrite("mana_regen_per_sec",    &VecSimulation::Config::manaRegenPerSec)
        .def_readwrite("damage",                &VecSimulation::Config::damage)
        .def_readwrite("attack_range",          &VecSimulation::Config::attackRange)
        .def_readwrite("attacks_per_sec",       &VecSimulation::Config::attacksPerSec)
        .def_readwrite("move_speed",            &VecSimulation::Config::moveSpeed)
        .def_readwrite("max_accel",             &VecSimulation::Config::maxAccel)
        .def_readwrite("max_turn_rate",         &VecSimulation::Config::maxTurnRate)
        .def_readwrite("radius",                &VecSimulation::Config::radius)
        .def_readwrite("reward_damage_dealt",     &VecSimulation::Config::rewardDamageDealt)
        .def_readwrite("reward_damage_taken_mul", &VecSimulation::Config::rewardDamageTakenMul)
        .def_readwrite("reward_kill",             &VecSimulation::Config::rewardKill)
        .def_readwrite("reward_death",            &VecSimulation::Config::rewardDeath)
        .def_readwrite("reward_step",             &VecSimulation::Config::rewardStep)
        .def_readwrite("reward_timeout",          &VecSimulation::Config::rewardTimeout);

    py::class_<VecSimulation>(m, "VecSimulation")
        .def(py::init<const VecSimulation::Config&>())
        .def_property_readonly("num_envs", &VecSimulation::numEnvs)
        .def_property_readonly("config",   &VecSimulation::config,
            py::return_value_policy::reference_internal)
        .def_readonly_static("HERO_ID",     &VecSimulation::HERO_ID)
        .def_readonly_static("OPPONENT_ID", &VecSimulation::OPPONENT_ID)

        .def("seed_and_reset", &VecSimulation::seedAndReset, py::arg("base_seed"))
        .def("reset_done",     &VecSimulation::resetDone)
        .def("reset_env",      &VecSimulation::resetEnv, py::arg("env_idx"))

        .def("observe", [](const VecSimulation& v, int agentId) {
            int N = v.numEnvs();
            py::array_t<float> out({N, observation::TOTAL});
            v.observe(agentId, out.mutable_data());
            return out;
        }, py::arg("agent_id"))

        .def("observe_into", [](const VecSimulation& v, int agentId,
                                py::array_t<float, py::array::c_style | py::array::forcecast> out) {
            if (out.ndim() != 2 || out.shape(0) != v.numEnvs()
                || out.shape(1) != observation::TOTAL) {
                throw py::value_error("observe_into: out must be (num_envs, OBS_TOTAL)");
            }
            v.observe(agentId, out.mutable_data());
        }, py::arg("agent_id"), py::arg("out"))

        .def("action_mask", [](const VecSimulation& v, int agentId) {
            int N = v.numEnvs();
            py::array_t<float> mask({N, action_mask::TOTAL});
            py::array_t<int>   ids({N, action_mask::N_ENEMY_SLOTS});
            v.actionMask(agentId, mask.mutable_data(), ids.mutable_data());
            return py::make_tuple(mask, ids);
        }, py::arg("agent_id"))

        .def("apply_actions", [](VecSimulation& v, int agentId,
                                 const std::vector<AgentAction>& actions) {
            if (static_cast<int>(actions.size()) != v.numEnvs()) {
                throw py::value_error("apply_actions: list length must equal num_envs");
            }
            v.applyActions(agentId, actions.data());
        }, py::arg("agent_id"), py::arg("actions"))

        // Vectorized path — avoids constructing N AgentAction Python objects
        // per tick. Callers pass numpy arrays directly; aim yaw/pitch default
        // to zero. Used by the self-play trainer to get ~3-4x throughput at
        // large NUM_ENVS.
        .def("apply_actions_raw", [](VecSimulation& v, int agentId,
                py::array_t<float,   py::array::c_style | py::array::forcecast> move,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> attack_target_id,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> use_ability_id) {
            int N = v.numEnvs();
            if (move.ndim() != 2 || move.shape(0) != N || move.shape(1) != 2)
                throw py::value_error("apply_actions_raw: move must be (num_envs, 2)");
            if (attack_target_id.ndim() != 1 || attack_target_id.shape(0) != N)
                throw py::value_error("apply_actions_raw: attack_target_id must be (num_envs,)");
            if (use_ability_id.ndim() != 1 || use_ability_id.shape(0) != N)
                throw py::value_error("apply_actions_raw: use_ability_id must be (num_envs,)");
            std::vector<AgentAction> acts(static_cast<size_t>(N));
            const float*   mv  = move.data();
            const int32_t* atk = attack_target_id.data();
            const int32_t* ab  = use_ability_id.data();
            for (int i = 0; i < N; i++) {
                acts[i].moveX          = mv[2 * i];
                acts[i].moveZ          = mv[2 * i + 1];
                acts[i].attackTargetId = atk[i];
                acts[i].useAbilityId   = ab[i];
            }
            v.applyActions(agentId, acts.data());
        }, py::arg("agent_id"), py::arg("move"),
           py::arg("attack_target_id"), py::arg("use_ability_id"))

        .def("step", &VecSimulation::step)

        .def("dones", [](const VecSimulation& v) {
            int N = v.numEnvs();
            py::array_t<int> done({N});
            py::array_t<int> winner({N});
            v.dones(done.mutable_data(), winner.mutable_data());
            return py::make_tuple(done, winner);
        })

        .def("rewards", [](VecSimulation& v) {
            int N = v.numEnvs();
            py::array_t<float> rh({N});
            py::array_t<float> ro({N});
            v.rewards(rh.mutable_data(), ro.mutable_data());
            return py::make_tuple(rh, ro);
        })

        .def("step_counts", [](const VecSimulation& v) {
            int N = v.numEnvs();
            py::array_t<int> out({N});
            v.stepCounts(out.mutable_data());
            return out;
        })

        .def("episode_counts", [](const VecSimulation& v) {
            int N = v.numEnvs();
            py::array_t<int> out({N});
            v.episodeCounts(out.mutable_data());
            return out;
        })

        .def("world", static_cast<World& (VecSimulation::*)(int)>(&VecSimulation::world),
             py::arg("env_idx"), py::return_value_policy::reference_internal)
        .def("hero",     &VecSimulation::hero,     py::arg("env_idx"),
             py::return_value_policy::reference_internal)
        .def("opponent", &VecSimulation::opponent, py::arg("env_idx"),
             py::return_value_policy::reference_internal);

    py::class_<ReplayReader>(m, "ReplayReader")
        .def(py::init<>())
        .def("open", &ReplayReader::open, py::arg("path"))
        .def_property_readonly("error_message", &ReplayReader::errorMessage)
        .def_property_readonly("header", &ReplayReader::header,
            py::return_value_policy::reference_internal)
        .def_property_readonly("roster", &ReplayReader::roster,
            py::return_value_policy::reference_internal)
        .def_property_readonly("frame_count", &ReplayReader::frameCount)
        .def("frame", &ReplayReader::frame, py::arg("index"))
        .def("find_by_step", &ReplayReader::findByStep, py::arg("step_idx"))
        .def("trajectory", &ReplayReader::trajectory, py::arg("agent_id"))
        .def("damage_summary", &ReplayReader::damageSummary);
}
