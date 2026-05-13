import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    # Скачиваем датасет
    url = 'https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv'
    df = pd.read_csv(url)
    df.to_csv("diamonds_raw.csv", index=False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    cat_columns = ['cut', 'color', 'clarity']
    
    # Очистка данных
    # Удаляем ошибки измерений
    question_dims = df[(df.x == 0) | (df.y == 0) | (df.z == 0)]
    df = df.drop(question_dims.index)
    
    # Удаляем слишком дорогие
    question_price = df[df.price > 15000]
    df = df.drop(question_price.index)
    
    df = df.reset_index(drop=True)  
    
    # Кодируем категориальные признаки
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    df_ordinal = pd.DataFrame(ordinal.transform(df[cat_columns]), columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    # Сохраняем чистый датасет
    df.to_csv('df_clear.csv', index=False)
    print("Data downloaded and cleared successfully!")
    return True

download_data()
clear_data("diamonds_raw.csv")