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

    // Every Value below is rooted in a Persistent before the next allocating
    // call (globalValue, getProperty, createObject, setProperty and
    // makeAiGameValue all may move the heap), and each allocating call's
    // argument is built in its own statement (embed.h's GC contract).
    ev::Persistent globalThisP;
    {
        auto gt = ev::globalValue("globalThis");
        if (gt.found && ev::isObject(gt.value)) globalThisP.set(gt.value);
    }
    const bool hasGlobalThis = ev::isObject(globalThisP.get());

    ev::Persistent broP;
    {
        auto b = ev::globalValue("bro");
        if (b.found && ev::isObject(b.value)) broP.set(b.value);
    }
    if (!ev::isObject(broP.get()) && hasGlobalThis) {
        Value candidate = ev::getProperty(globalThisP.get(), "bro");
        if (ev::isObject(candidate)) broP.set(candidate);
    }
    if (!ev::isObject(broP.get())) {
        broP.set(ev::createObject());
        ev::registerGlobal("bro", broP.get());
        if (hasGlobalThis) {
            globalThisP.set(ev::setProperty(globalThisP.get(), "bro", broP.get()));
        }
    }

    ev::Persistent aiP(ev::getProperty(broP.get(), "ai"));
    if (!ev::isObject(aiP.get())) {
        aiP.set(ev::createObject());
        broP.set(ev::setProperty(broP.get(), "ai", aiP.get()));
    }

    ev::Persistent gameP(makeAiGameValue());
    aiP.set(ev::setProperty(aiP.get(), "game", gameP.get()));

    ev::registerGlobal("AI", gameP.get());
    if (hasGlobalThis) {
        globalThisP.set(ev::setProperty(globalThisP.get(), "AI", gameP.get()));
    }
}

} // namespace brogameagent::api
