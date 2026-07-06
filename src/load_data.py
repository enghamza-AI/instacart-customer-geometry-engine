# load_data.py

import pandas as pd
import numpy as np
import yaml
import time
import warnings




def load_config(config_path: str = "config/config.yaml") -> dict:
  
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_and_merge(config: dict) -> pd.DataFrame:

    verbose = config["data"]["verbose"]

    if verbose:
        print("=" * 60)
        print("LOADING & MERGING INSTACART DATA")
        print("=" * 60)

    
    if verbose:
        print("\n[1/5] Loading order_products__prior.csv (~32M rows) ...")
    t0 = time.time()

    order_products = pd.read_csv(
        config["data"]["order_products_path"],
        usecols=["order_id", "product_id", "reordered"],
        dtype={
            "order_id":   "int32",
            "product_id": "int32",
            "reordered":  "int8",
        }
    )

    if verbose:
        mem = order_products.memory_usage(deep=True).sum() / 1e6
        print(f"   ✓ {order_products.shape[0]:,} rows | {order_products.shape[1]} cols | {mem:.0f} MB | {time.time()-t0:.0f}s")

   
    if verbose:
        print("\n[2/5] Loading orders.csv ...")
    t0 = time.time()

    orders = pd.read_csv(
        config["data"]["orders_path"],
        dtype={
            "order_id":          "int32",
            "user_id":           "int32",
            "order_number":      "int16",
            "order_dow":         "int8",
            "order_hour_of_day": "int8",
            
        }
    )

    if verbose:
        print(f"   ✓ {orders.shape[0]:,} rows | {orders['user_id'].nunique():,} unique users")
        print(f"   eval_set distribution: {dict(orders['eval_set'].value_counts())}")

  
    if verbose:
        print("\n[3/5] Loading products, aisles, departments ...")

    products = pd.read_csv(
        config["data"]["products_path"],
        dtype={"product_id": "int32", "aisle_id": "int16", "department_id": "int8"}
    )
    aisles      = pd.read_csv(config["data"]["aisles_path"])
    departments = pd.read_csv(config["data"]["departments_path"])

    if verbose:
        print(f"   ✓ products: {products.shape[0]:,} rows")
        print(f"   ✓ aisles: {aisles.shape[0]} rows | departments: {departments.shape[0]} rows")

  
    if verbose:
        print("\n[4/5] Merging tables ...")
    t0 = time.time()

    
    df = order_products.merge(orders, on="order_id", how="left")
    if verbose:
        print(f"   After orders merge: {df.shape}")

   
    df = df.merge(products, on="product_id", how="left")
    if verbose:
        print(f"   After products merge: {df.shape}")

    
    df = df.merge(departments, on="department_id", how="left")

    
    df = df.merge(aisles, on="aisle_id", how="left")

    if verbose:
        mem = df.memory_usage(deep=True).sum() / 1e6
        print(f"   Final merged shape: {df.shape} | {mem:.0f} MB | {time.time()-t0:.0f}s")

 
    if verbose:
        print("\n[5/5] Filtering to prior orders only ...")

    before = len(df)
    df = df[df["eval_set"] == "prior"].copy()
    df = df.drop(columns=["eval_set"])   
    after = len(df)

    if verbose:
        print(f"   Rows: {before:,} → {after:,} ({before-after:,} non-prior rows dropped)")
        print(f"   Unique users: {df['user_id'].nunique():,}")
        print(f"   Unique products: {df['product_id'].nunique():,}")
        print(f"\n✓ Load + merge complete. Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

    return df




def sample_users(df: pd.DataFrame, config: dict) -> pd.DataFrame:


    sample_size = config["data"]["sample_size"]
    random_seed = config["data"]["random_seed"]
    verbose     = config["data"]["verbose"]

    all_users   = df["user_id"].unique()
    total_users = len(all_users)

    if verbose:
        print(f"\nSampling {sample_size:,} users from {total_users:,} total ...")

    
    if sample_size >= total_users:
        if verbose:
            print(f"  sample_size ≥ total users — using full dataset.")
        return df

    
    np.random.seed(random_seed)

    sampled_user_ids = np.random.choice(
        all_users,
        size=sample_size,
        replace=False   
                        
    )

    # isin() is the pandas equivalent of SQL's WHERE user_id IN (...)
    df_sampled = df[df["user_id"].isin(sampled_user_ids)].copy()

    if verbose:
        print(f"  ✓ Sampled users: {df_sampled['user_id'].nunique():,}")
        print(f"  ✓ Rows retained: {len(df_sampled):,} of {len(df):,} ({len(df_sampled)/len(df)*100:.1f}%)")

    return df_sampled




def check_representativeness(
    df_full:    pd.DataFrame,
    df_sampled: pd.DataFrame,
    verbose:    bool = True
) -> dict:

    
    reorder_full   = df_full["reordered"].mean()
    reorder_sample = df_sampled["reordered"].mean()

   
    orders_full   = df_full.groupby("user_id")["order_id"].nunique().mean()
    orders_sample = df_sampled.groupby("user_id")["order_id"].nunique().mean()

   
    dept_full   = df_full["department"].nunique()
    dept_sample = df_sampled["department"].nunique()

    stats = {
        "reorder_rate_full":          round(float(reorder_full), 4),
        "reorder_rate_sample":        round(float(reorder_sample), 4),
        "avg_orders_per_user_full":   round(float(orders_full), 2),
        "avg_orders_per_user_sample": round(float(orders_sample), 2),
        "dept_coverage_full":         int(dept_full),
        "dept_coverage_sample":       int(dept_sample),
    }

    if verbose:
        print("\n--- REPRESENTATIVENESS CHECK ---")
        print(f"  Reorder rate:       full={reorder_full:.3f}  |  sample={reorder_sample:.3f}", end="  ")
        diff = abs(reorder_full - reorder_sample)
        print("✓" if diff < 0.02 else f" diff={diff:.3f} — consider different random_seed")

        print(f"  Avg orders/user:    full={orders_full:.1f}  |  sample={orders_sample:.1f}")
        print(f"  Dept coverage:      full={dept_full}  |  sample={dept_sample}", end="  ")
        print("✓" if dept_sample == dept_full else f" {dept_full - dept_sample} dept(s) missing from sample")

    return stats




def health_check(df: pd.DataFrame, verbose: bool = True) -> bool:

    if verbose:
        print("\n--- HEALTH CHECK ---")

    issues = []

   
    for col in ["user_id", "order_id", "product_id", "reordered"]:
        if col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                issues.append(f"{col} has {nulls:,} unexpected nulls")
            elif verbose:
                print(f"  ✓ {col}: no nulls")

    
    unique_vals = set(df["reordered"].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        issues.append(f"'reordered' has non-binary values: {unique_vals}")
    elif verbose:
        print(f"   reordered: strictly binary (0/1)")

   
    dept_count = df["department"].nunique()
    if dept_count < 21:
        issues.append(f"Only {dept_count}/21 departments present — binary flags may be incomplete")
    elif verbose:
        print(f"   All {dept_count} departments present")


    n_dupes = df.duplicated(subset=["user_id", "order_id", "product_id"]).sum()
    if n_dupes > 0:
        issues.append(f"{n_dupes:,} duplicate (user, order, product) rows found")
    elif verbose:
        print(f"   No duplicate (user, order, product) rows")

    if issues:
        print("\n   HEALTH CHECK WARNINGS:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        if verbose:
            print(f"\n   All health checks passed.")
        return True



def sample_and_validate(df_full: pd.DataFrame, config: dict) -> pd.DataFrame:

    verbose = config["data"]["verbose"]

    if verbose:
        print("\n" + "=" * 60)
        print("SAMPLING & VALIDATION")
        print("=" * 60)

    
    df_sampled = sample_users(df_full, config)

   
    check_representativeness(df_full, df_sampled, verbose=verbose)

   
    passed = health_check(df_sampled, verbose=verbose)

    if not passed:
        warnings.warn(
            "Health check found issues — review warnings above before proceeding. "
            "Features built on unhealthy data may produce incorrect distance metrics.",
            UserWarning
        )
    else:
        if verbose:
            n_users = df_sampled["user_id"].nunique()
            print(f"\n✓ Sample ready: {n_users:,} customers → features.py")

    return df_sampled




def run_full_pipeline(config: dict) -> pd.DataFrame:


   
    df_full = load_and_merge(config)

 
    df_ready = sample_and_validate(df_full, config)

    return df_ready


if __name__ == "__main__":

    print("Running load_data.py standalone ...\n")

    config   = load_config()
    df_ready = run_full_pipeline(config)

    print("\n--- FINAL OUTPUT ---")
    print(f"Shape:   {df_ready.shape}")
    print(f"Columns: {list(df_ready.columns)}")
    print(f"Users:   {df_ready['user_id'].nunique():,}")
    print(f"Memory:  {df_ready.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(f"\nSample rows:")
    print(df_ready.head(5).to_string())
    print(f"\nNull counts:")
    print(df_ready.isnull().sum())
    print(f"\ndtypes:")
    print(df_ready.dtypes)
