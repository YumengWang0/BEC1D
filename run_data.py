import datetime
from libs.data_gene import *

def main():

    RANDOM_SEED = 20260425

    PHYSICS_DOMAIN = {
        "x_a": -20,
        "x_b":  20,
        "N":  4096,
        "x0":    0,
    }

    TRAIN_CONFIG = {
        "mu_left":  -0.2222,
        "mu_right": -0.05,
        "N_mu":      15,
    }

    TEST_CONFIG = {
        "mu_left":  -0.2222,
        "mu_right": -0.05,
        "delta_max":  1.1,
        "N_test":     120,
        "seed":    RANDOM_SEED,
    }

    eq_name  = "Ex1"
    sol_name = f"./data/solution_{eq_name}.h5"

    train_pairs = build_train_pairs(TRAIN_CONFIG)
    test_pairs  = build_test_pairs(TEST_CONFIG)

    print(f"Train pairs : {len(train_pairs)}")
    print(f"Test  pairs : {len(test_pairs)}")

    train_data = build_dataset(train_pairs, PHYSICS_DOMAIN)
    test_data  = build_dataset(test_pairs,  PHYSICS_DOMAIN)

    print(f"Train phi shape : {train_data['phi'].shape}")
    print(f"Test  phi shape : {test_data['phi'].shape}")

    now = datetime.datetime.today()
    save_hdf5(sol_name, train_data, test_data, now )
    print(f"Saved → {sol_name}")


if __name__ == "__main__":
    main()