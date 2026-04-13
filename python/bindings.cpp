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
}
