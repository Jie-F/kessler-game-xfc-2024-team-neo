from src.kesslergame import KesslerController

class NullController(KesslerController):
    def __init__(self):
        pass

    def actions(self, ship_state: dict, game_state: dict) -> tuple[float, float, bool, bool]:

        thrust = 0
        turn_rate = 0
        fire = False
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Null Controller"
