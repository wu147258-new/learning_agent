from dataclasses import dataclass

ACTIONS = {"inspect", "patch_syntax", "patch_logic", "test"}

@dataclass
class World:
    fault: str
    diagnosed: bool = False
    repaired: bool = False
    passed: bool = False

    def execute(self, action: str) -> str:
        if action not in ACTIONS:
            return "DENIED: unknown action"
        if action == "inspect":
            self.diagnosed = True
            return f"OBSERVE: fault={self.fault}"
        if action.startswith("patch_"):
            if not self.diagnosed:
                return "ERROR: inspect first"
            expected = "patch_" + self.fault
            self.repaired = action == expected
            return "PATCHED" if self.repaired else "WRONG PATCH"
        self.passed = bool(self.repaired)
        return "PASS" if self.passed else "FAIL"

def choose_action(observation: str) -> str:
    if observation == "START":
        return "inspect"
    if observation.startswith("OBSERVE: fault="):
        return "patch_" + observation.split("=", 1)[1]
    if observation == "PATCHED":
        return "test"
    return "STOP"

def run(fault: str, max_steps: int = 6):
    world = World(fault=fault)
    trace, observation = [], "START"
    for i in range(max_steps):
        action = choose_action(observation)
        if action == "STOP":
            break
        observation = world.execute(action)
        trace.append((i, action, observation))
        if observation == "PASS":
            break
    return world.passed, trace

if __name__ == "__main__":
    for fault in ("syntax", "logic"):
        ok, trace = run(fault)
        print(fault, ok, trace)
        assert ok
