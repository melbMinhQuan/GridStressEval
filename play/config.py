"""Model inputs shared by every play script. Math: play/allocation_math.pdf."""
from pathlib import Path

N = 40                                            # customers per circuit
T = 1000                                          # circuit current limit, A
CURRENT = {"S": 20, "M": 40, "L": 100}            # A per customer
V = 240                                           # V; load P_j = V * I_j
P_TYPE = {"S": 0.69, "M": 0.30, "L": 0.01}        # customer type probabilities
DEMAND_SCALE = 0.1                                # scales demand used in SOAK only (overload uses full current)

INTERVAL_SETS = {                                 # hours, decimals allowed (10.5 = 10:30)
    1: [(11, 14)],
    2: [(8, 11), (11, 14)],
    3: [(8, 11), (10.5, 13.5), (11, 14)],
}

UPTAKE = 0.1                                      # share of VIC customers on the tariff
VIC_CUSTOMERS = 2_900_000                         # approx. VIC electricity customers

SOLAR_SOURCE = "curtailment"                      # "curtailment", "rooftop" (shape), "rooftop_actual", "rooftop_local", "rooftop_5m"
LOCAL_TIME = False                                # True: interval times are local clock time (DST aware)

ROOT = Path(__file__).resolve().parents[1]
SOLAR_CSV = ROOT / "script" / "data" / "NEM_VIC1_curtailment_2025_5m.csv"
ROOFTOP_DEMAND_CSV = ROOT / "script" / "data" / "NEM_VIC1_rooftop_demand_2025_1h.csv"
ROOFTOP_5M_CSV = ROOT / "script" / "data" / "NEM_VIC1_rooftop_2025_5m.csv"
DST_END, DST_START = "2025-04-06", "2025-10-05"   # Victoria 2025: AEDT (UTC+11) before END and from START
OUT_DIR = Path(__file__).resolve().parent / "out"
SLOTS = 288                                       # 5-minute slots per day
DT = 24 / SLOTS                                   # hours per slot


def load_kw():
    """P_j = V * I_j in kW per customer."""
    return {j: V * amps / 1000 for j, amps in CURRENT.items()}


def circuits(uptake=None):
    """Number of participating circuits C = uptake * VIC_CUSTOMERS / N."""
    return (UPTAKE if uptake is None else uptake) * VIC_CUSTOMERS / N
