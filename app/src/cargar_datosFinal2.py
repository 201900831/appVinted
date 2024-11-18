#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

import pandas as pd

def load_product_data_from_csv(file_path):
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        #print("Datos cargados correctamente desde el archivo CSV.")
        return df
    except Exception as e:
        print("Error al cargar los datos desde el archivo CSV:", e)
        return pd.DataFrame()



# In[9]:


def prepare_data_for_model(df):
    """
    Prepare data for modeling by handling missing values and encoding categorical features.
    """
    # Selecting relevant columns for prediction
    relevant_columns = [
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
        "PageValues", "SpecialDay", "Month", "OperatingSystems", "Browser",
        "Region", "TrafficType", "VisitorType", "Weekend", "Revenue"
    ]
    df = df[relevant_columns]
    # Rename 'Revenue' to 'will_buy' for consistency with classification purpose
    df = df.rename(columns={"Revenue": "will_buy"})
    
    # Fill missing values if necessary (there shouldn’t be any, but just in case)
    df = df.fillna(0)

    # Convert categorical columns to the appropriate format
    df["VisitorType"] = df["VisitorType"].astype("category")
    df["Month"] = df["Month"].astype("category")
    df["Weekend"] = df["Weekend"].astype("int")  # Weekend is likely a binary indicator (0 or 1)
    df["will_buy"] = df["will_buy"].astype("int")  # Ensure target variable is integer format

    return df


# In[10]:


def load_and_prepare_data(file_path):
    """
    Load data from a CSV file and prepare it for modeling.
    """
    # Load the data
    df = load_product_data_from_csv(file_path)
    
    # Prepare the data
    df_prepared = prepare_data_for_model(df)
    return df_prepared


# In[11]:


if __name__ == "__main__":
    # Replace with the path to your actual CSV file
    df_products = load_and_prepare_data("online_shoppers_intention.csv")
    print(df_products.head())  # Display the first rows of the DataFrame

