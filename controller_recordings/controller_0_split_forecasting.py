
from typing import Dict, Tuple

class ReplayController0:
    def __init__(self):
        self.recorded_actions = [(0.0, 141.78804544559526, False, False), (480.0, -80.23292522852765, False, False), (480.0, -80.23292522852765, False, False), (480.0, -80.23292522852765, False, False), (480.0, -80.23292522852765, False, False), (480.0, -80.23292522852765, False, False), (480.0, -80.23292522852765, False, False), (480.0, -80.23292522852765, False, False), (480.0, -80.23292522852765, False, False), (480.0, -80.23292522852765, True, False), (263.55594277429077, 151.55783429601405, False, False), (80.0, 18.78755924491853, False, False), (80.0, 84.55544420492063, True, False), (80.0, -28.626476678360365, False, False), (-480.0, 0.0, False, False), (-480.0, 0.0, True, False), (-480.0, 96.57735803376343, False, False), (-480.0, -0.40267655770285565, False, False), (-480.0, 0.0, True, False), (-480.0, 0.0, False, False), (-423.5559427742901, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False), (0.0, 0.0, False, False)]
        self.current_step = 0

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        if self.current_step < len(self.recorded_actions):
            action = self.recorded_actions[self.current_step]
            self.current_step += 1
            return tuple(action)
        else:
            return 0.0, 0.0, False, False  # Default action if out of recorded actions

    @property
    def name(self) -> str:
        return "Neo Replay Controller"
            