from src.kesslergame import KesslerController

class BenchmarkController(KesslerController):
    def __init__(self) -> None:
        pass

    def actions(self, ship_state: dict, game_state: dict) -> tuple[float, float, bool, bool]:

        thrust = 400
        turn_rate = 6
        fire = True
        drop_mine = True

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Benchmark Controller"
