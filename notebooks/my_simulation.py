import pandas as pd
import re

import re


def format_field(name):
    name = name.lower()
    name = name.replace("°", "")
    name = name.replace("·", "")
    name = name.replace("/", "")
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


class VesselEnvironment:
    def __init__(self):
        self.vessel_diameter_model = None
        self.rcylrves_aorta_small_umr = None
        self.rcylrves_aorta_large_umr = None
        self.rcylrves_renal_small_umr = None
        self.rcylrves_renal_large_umr = None
        self.renal_bifurcation_angle = None
        self.friction_coeff_aorta_small_umr = None
        self.friction_coeff_aorta_large_umr = None
        self.friction_coeff_renal_small_umr = None


class UMRProperties:
    def __init__(self):
        self.umr_length = None
        self.umr_diameter = None
        self.aortic_flow_rate = None
        self.pulsatile_flow_range = None
        self.max_upstream_flow = None
        self.umr_speed_9_hz = None
        self.re_9mm_1hz_25c = None
        self.re_9mm_9hz_25c = None
        self.re_12mm_1hz_25c = None
        self.re_12mm_9hz_25c = None
        self.re_9mm_1hz_37c = None
        self.re_9mm_9hz_37c = None
        self.re_12mm_1hz_37c = None
        self.re_12mm_9hz_37c = None
        self.viscosity_at_25c = None
        self.viscosity_at_37c = None
        self.beta_blood = None
        self.beta_clot = None
        self.deborah_number = None
        self.physiological_temp = None
        self.speed_in_clots = None


def main():
    csv_path = "UMR_Parameters.csv"

    vessel = VesselEnvironment()
    umr = UMRProperties()

    df = pd.read_csv(csv_path, encoding="utf-8")

    for _, row in df.iterrows():
        param = format_field(row["Parameter"])
        value = row["Typical Value"]
        if hasattr(vessel, param):
            setattr(vessel, param, value)
        elif hasattr(umr, param):
            setattr(umr, param, value)
        else:
            print(f"⚠️ Unmatched parameter: {param}")

    print("\n=== Vessel Environment Parameters ===")
    for key, value in vars(vessel).items():
        print(f"{key}: {value}")

    print("\n=== UMR Properties ===")
    for key, value in vars(umr).items():
        print(f"{key}: {value}")

    print("\nFormatted parameter names from CSV:")
    for _, row in df.iterrows():
        param = format_field(row["Parameter"])
        print(param)

    print(vars(vessel).keys())


if __name__ == "__main__":
    main()
