import datetime
from libs.data_gene import *
import libs.config as config
def main():
 

    eq_name  = "Ex1"
    sol_name = f"./data/solution_{eq_name}.h5"

    train_pairs = build_train_pairs(config.TRAIN_CONFIG)
    test_pairs  = build_test_pairs(config.TEST_CONFIG)

    print(f"Train pairs : {len(train_pairs)}")
    print(f"Test  pairs : {len(test_pairs)}")

    train_data = build_dataset(train_pairs, config.PHYSICS_DOMAIN)
    test_data  = build_dataset(test_pairs,  config.PHYSICS_DOMAIN)

    print(f"Train phi shape : {train_data['phi'].shape}")
    print(f"Test  phi shape : {test_data['phi'].shape}")

    now = datetime.datetime.today()
    save_hdf5(sol_name, train_data, test_data, now )
    print(f"Saved → {config.sol_name}")


if __name__ == "__main__":
    main()