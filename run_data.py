import datetime
from libs.data_gene import *
import libs.config as config
import os 


def main():
 

 
    if  not os.path.exists(config.config["sol_name"]):

        print(f"[DATA] {config.config['sol_name']} not found → generating...")

        now = datetime.datetime.today()
        train_pairs = build_train_pairs(config.TRAIN_CONFIG)
        test_pairs  = build_test_pairs(config.TEST_CONFIG)

        print(f"Train pairs : {len(train_pairs)}")
        print(f"Test  pairs : {len(test_pairs)}")

        train_data = build_dataset(train_pairs, config.PHYSICS_DOMAIN)
        test_data  = build_dataset(test_pairs,  config.PHYSICS_DOMAIN)

        print(f"Train phi shape : {train_data['phi'].shape}")
        print(f"Test  phi shape : {test_data['phi'].shape}")

   
        save_hdf5(config.config["sol_name"], train_data, test_data, now )
        print(f"Saved → {config.config['sol_name']}")

    else: 

        print(f"[DATA] Found {config['sol_name']} → skip generation")




if __name__ == "__main__":
    main()
