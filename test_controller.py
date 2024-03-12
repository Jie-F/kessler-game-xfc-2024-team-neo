from src.kesslergame import KesslerController

class TestController(KesslerController):
    def __init__(self) -> None:
        self.ts = -1

    def actions(self, ship_state: dict, game_state: dict) -> tuple[float, float, bool, bool]:
        self.ts += 1
        print(f"TS: {self.ts}, can shoot: {ship_state['can_fire']}, can lay mine: {ship_state['can_deploy_mine']}")
        if self.ts == 0:
            return -100, 0, True, False
        elif self.ts == 7:
            return -100, 0, True, False
        else:
            return -100, 0, False, False

    @property
    def name(self) -> str:
        return "Test Controller"
