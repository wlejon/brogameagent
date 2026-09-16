#include "api.h"
#include "host_ai_internal.h"

namespace brogameagent::api {

static NavMeshHooks s_navMeshHooks;

void setNavMeshHooks(const NavMeshHooks& hooks) {
    s_navMeshHooks = hooks;
}

const NavMeshHooks& getNavMeshHooks() {
    return s_navMeshHooks;
}

void installGameAi() {
    ensureAIClassesInstalled();
    ensureAIMctsClassesInstalled();

    Value globalThisVal = ev::undefined();
    auto gt = ev::globalValue("globalThis");
    if (gt.found && ev::isObject(gt.value)) {
        globalThisVal = gt.value;
    }

    Value broVal = ev::globalValue("bro").found ? ev::globalValue("bro").value : ev::undefined();
    if (!ev::isObject(broVal)) {
        if (!ev::isUndefined(globalThisVal)) {
            Value candidate = ev::getProperty(globalThisVal, "bro");
            if (ev::isObject(candidate)) {
                broVal = candidate;
            }
        }
    }
    if (!ev::isObject(broVal)) {
        broVal = ev::createObject();
        ev::registerGlobal("bro", broVal);
        if (!ev::isUndefined(globalThisVal)) {
            ev::setProperty(globalThisVal, "bro", broVal);
        }
    }

    ev::Persistent broP(broVal);

    Value aiVal = ev::getProperty(broP.get(), "ai");
    if (!ev::isObject(aiVal)) {
        aiVal = ev::createObject();
        broP.set(ev::setProperty(broP.get(), "ai", aiVal));
    }

    ev::Persistent aiP(aiVal);
    Value gameVal = makeAiGameValue();
    aiP.set(ev::setProperty(aiP.get(), "game", gameVal));
}

} // namespace brogameagent::api
