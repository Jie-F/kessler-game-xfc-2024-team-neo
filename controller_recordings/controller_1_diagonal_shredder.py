
from typing import Dict, Tuple

class ReplayController1:
    def __init__(self):
        self.recorded_actions = [(92.97793916624401, -180.0, False, False), (83.94155750983273, -180.0, False, False), (40.26637891507392, -180.0, False, False), (-89.42578957687446, -180.0, False, False), (-134.04320447736225, -180.0, False, False), (-147.75854416169784, -180.0, False, False), (-146.83373475667176, -180.0, False, False), (-123.6375938709164, -180.0, False, False), (-105.81887237667482, -180.0, False, False), (-152.48239316804117, -180.0, False, False), (-124.30498557105967, -180.0, False, False), (58.69768144706002, -180.0, False, False), (105.42862519204081, -180.0, False, False), (119.00858322697945, -180.0, False, False), (109.2918030539341, -180.0, False, False), (79.77237292059762, 180.0, False, False), (84.37182318067394, 180.0, False, False), (81.03480210019697, 180.0, False, False), (79.54465891760218, 180.0, False, False), (80.47684072300457, 180.0, False, False), (80.01678065104142, 180.0, False, False), (78.9544521101108, 180.0, False, False), (78.4405032407251, 180.0, False, False), (78.37194028550871, 169.1909807034223, False, False), (78.94429800732365, -34.83752423908107, False, False), (78.93338939925789, -32.61000751759554, False, False), (79.3653850144421, -31.355474053015268, False, False), (79.6401242244226, -30.1126125187384, False, False), (79.81157419529336, -28.912136772070728, False, False), (79.91808444617223, -27.75350723487532, False, False), (79.9833328499104, -26.636551815255142, False, False), (80.01933501627119, -25.561249896165723, False, False), (79.56797870295027, -24.52760135932859, False, False), (79.83594967104256, -23.54901573375153, False, False), (80.01337059336407, -22.58951581874043, False, False), (80.10494186934807, -180.0, False, False), (79.99935505166566, -180.0, False, False), (79.92623355193776, -180.0, False, False), (79.87585629901044, -180.0, False, False), (79.84553304516677, -180.0, False, False), (79.88917200920216, -180.0, False, False), (79.99125938406294, -180.0, False, False), (76.51906493406457, -127.24055179651369, False, False), (83.60670109527551, 2.748513067171768, False, False), (87.82780615468809, 1.905274138053399, False, False), (85.99171644329266, 1.8220356708564551, False, False), (84.5080090487065, 1.7533317896831884, False, False), (82.00532340052104, 1.687781104728695, False, False), (80.22358930361871, 1.6279881592454712, False, False), (73.22687195182903, 1.5700467638123776, False, False), (110.17750753586996, 1.5249753294704125, False, False), (92.00625284195186, 1.4028062094297709, False, False), (-77.8205357599264, 1.384961245963198, False, False), (-145.0909031373216, 1.6194805241397183, False, False), (-105.84510428372603, 1.5458006883008755, False, False), (-96.65063194852492, 106.00017221725773, False, False), (-90.26937430639013, 12.647732374207388, False, False), (-87.38351054408763, 11.426033566306275, False, False), (-88.07035751216756, 10.78627279315625, False, False), (-99.36773906549871, 10.24158924848195, False, False), (-114.63327336973153, 9.859648280064818, False, False), (-130.73859971569385, 9.542932862831975, False, False), (-141.27971459502555, 9.248918606185256, False, False), (-146.1087196294011, 8.911054281694655, False, False), (-154.74822770338463, 8.53612964848112, False, False), (-144.45766611879725, 180.0, False, False), (-149.86775811270027, 180.0, False, False), (-115.87100310537124, 180.0, False, False), (-95.9556200470249, -128.49973572949125, False, False), (-92.88187745487865, -180.0, False, False), (-137.807689812656, -180.0, False, False), (-146.22940050805167, -180.0, False, False), (-135.19911523954337, -180.0, False, False), (-133.02099445281397, -180.0, False, False), (-133.52985137782431, 180.0, False, False), (-106.64113610112669, 129.90237814246734, False, False), (-73.42470512333041, 4.465499256239445, False, False), (-57.65350783151302, 0.7414592660715135, False, False), (-40.48218243021665, 0.6036423197918599, False, False), (73.18775201963716, 0.5727257358560588, False, False), (105.50350086545056, 0.4795300954468631, False, False), (121.68528501590886, 0.4422246110755571, False, False), (132.0089823235961, 0.41831956298328593, False, False), (121.64138098526352, -180.0, False, False), (116.89365584375658, -180.0, False, False), (116.3150225783028, -180.0, False, False), (108.23074084845834, -180.0, False, False), (107.60961181580178, -180.0, False, False), (106.89091236271676, -180.0, False, False), (-158.1963245501495, 180.0, False, False), (-157.48049218061703, 180.0, False, False), (-107.49015074691685, 180.0, False, False), (-99.73532484043393, 180.0, False, False), (-92.89309095054783, -180.0, True, False), (-98.03252087171808, -180.0, False, False), (-108.40603127715521, -180.0, False, False), (-120.0225506521985, -180.0, False, False), (98.19291326834147, -180.0, False, False), (76.02644798151445, -180.0, False, False), (48.052754320980036, -180.0, False, False), (-78.3474179247776, -180.0, False, False), (-142.36295092246448, -180.0, False, False), (-131.28642128449326, -180.0, False, False), (-93.20698409792467, -180.0, False, False), (-86.69394808233, -180.0, False, False), (-85.97511973834175, 180.0, False, False), (-158.21560434207566, -180.0, False, False), (-97.35760086732725, 61.95946041513843, False, False), (-94.14832545817265, 180.0, False, False), (-90.86565575325751, 180.0, False, False), (-88.5229781669833, 180.0, False, False), (-86.7833781593584, 180.0, False, False), (-85.4229133662641, 180.0, False, False), (-84.32530030996854, 180.0, False, False), (-83.41553274400007, 180.0, False, False), (-82.64002163965407, 180.0, False, False), (-81.91039421817229, 180.0, False, False), (-81.34178166138167, 180.0, False, False), (-80.76180163207609, 138.70850928552005, False, False), (-80.18738177150367, 180.0, False, False), (-75.20689138301026, 180.0, False, False), (-78.64666262354075, 180.0, False, False), (-80.68074766343103, 180.0, False, False), (-80.63847449797689, 180.0, False, False), (-81.37375494691493, 180.0, False, False), (-82.95868823463896, 180.0, False, False), (-88.2054930869525, 180.0, False, False), (-94.17554895983716, 180.0, False, False), (-100.7605369417361, -180.0, False, False), (-91.05574970892665, -180.0, False, False), (-84.2033099862483, -180.0, False, False), (-89.9348997584383, -180.0, False, False), (-120.18377535002816, -180.0, False, False), (-140.4505299663812, 180.0, False, False), (-142.55893206590935, 180.0, False, False), (-128.12295427367613, 180.0, False, False), (-100.13856049106253, 180.0, False, False), (-3.681908247889868, 180.0, False, False), (89.1347414953834, 180.0, False, False), (99.49090189018703, 180.0, False, False), (107.04309101918267, 180.0, False, False), (115.3281460874155, 180.0, False, False), (74.34061418477256, 180.0, False, False), (-130.57132087917915, 180.0, False, False), (-119.92542303578152, 180.0, False, False), (-101.59739362502812, 47.902810941886145, False, False), (-95.25446920300088, 40.02871481929996, False, False), (-91.75822085852059, 69.89379362076315, False, False), (-90.53319178172951, 168.51287125469392, False, False), (-91.96647194884608, 180.0, False, False), (106.92962962590059, 180.0, False, False), (116.86338395570047, 180.0, False, False), (110.34082800707938, 180.0, False, False), (115.57053974793905, 180.0, False, False), (110.89195047032652, 180.0, False, False), (106.23595945623079, 180.0, False, False), (68.17147528528821, 180.0, False, False), (46.88432691740403, -180.0, False, False), (58.598850294496685, -180.0, False, False), (87.045651381237, -180.0, False, False), (118.6585637457074, -180.0, False, False), (111.33321134513912, -180.0, False, False), (25.23347423675396, -180.0, False, False), (63.554638691112444, -180.0, False, False), (97.19087144638057, -180.0, False, False), (117.6258895693711, -180.0, False, False), (67.7839326365531, -180.0, False, False), (52.7805239245576, -180.0, False, False), (51.8761500747547, -180.0, False, False), (56.620251935405314, -180.0, False, False), (63.85246743665297, 180.0, False, False), (70.82067307378729, 180.0, False, False), (73.99260001356775, 180.0, False, False), (72.61144573565497, 180.0, False, False), (-158.0840185788451, 158.76326463283903, False, False), (-107.91211378590926, -180.0, False, False), (-139.17985592014128, -180.0, False, False), (-155.14628781650256, -180.0, False, False), (-151.53851801758532, -180.0, False, False), (-152.7719735797334, -180.0, False, False), (-153.17756982816397, -180.0, False, False), (-141.9094006129604, -180.0, False, False), (-130.29281143885444, -180.0, False, False), (-143.49105208694192, -180.0, False, False), (67.39035584390606, -180.0, False, False), (-120.14965022078148, -51.14828880277076, False, False), (-153.8374223445207, -4.524544036806119, False, False), (-136.95559370445855, -3.883252483628933, False, False), (66.49312703614727, -3.7196202788097095, False, False), (102.67636801321306, -2.81022252032805, False, False), (125.72416482056329, -2.576969501507971, False, False), (130.07078957865093, -2.414314612665528, False, False), (113.10964698537437, -2.3309434089251835, False, False), (78.45912147640855, -2.7754610307228567, False, False), (87.06573445430556, -3.028785807082309, False, False), (83.68363774024351, -2.7627038776868957, False, False), (81.51481836588202, -6.187912799767889, False, False), (79.61251004656698, -5.941998134906508, False, False), (76.81083360720417, -5.717256103058918, True, False), (75.13241749486022, -5.511241298408861, False, False), (77.132104181673, -5.308896585673025, False, False), (81.59140836440979, -5.0940934729611165, False, False), (83.34833495082809, -4.876321600762948, False, False), (82.48274504806919, -4.68694970522926, False, False), (80.49959509801195, -4.522789311288809, True, False), (79.22221912068132, -4.373213860933752, False, False), (78.54012910436444, -4.227321171566355, False, False), (78.27719230952489, 154.50824116063, False, False), (78.95147274082372, -50.64049297424452, False, False), (78.5742598959968, -48.38015041655514, False, False), (78.55644707486952, -71.17786804911664, True, False), (78.66750888429314, 180.0, False, False), (78.85494829105399, 180.0, False, False), (79.13281864608922, 180.0, False, False), (79.8744003792694, 180.0, False, False), (80.84417343465628, 180.0, False, False), (82.04123406319208, 180.0, True, False), (83.27999594491892, 180.0, False, False), (83.98356836987536, 180.0, False, False), (84.07923125839925, 121.1546210055202, False, False), (83.59472844369631, -29.42722901254081, False, False), (72.95615082995062, -27.96338243042012, False, False), (111.6045818861552, -27.91807497012761, False, False), (91.66212274665693, -26.184500213421764, True, False), (-95.43281802366168, -26.42657854842553, False, False), (-148.67364424711693, 180.0, False, False), (-154.83960976006338, 180.0, False, False), (-149.88292341475008, 180.0, False, False), (-109.38882625033571, 180.0, False, False), (-112.64340403392859, 180.0, True, False), (-134.34521426429598, 180.0, False, False)]
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
        return "Baby Neo Replay Controller"
            