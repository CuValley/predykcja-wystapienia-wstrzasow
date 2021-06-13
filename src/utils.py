import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tqdm

from sklearn.ensemble import RandomForestClassifier


def add_strength(energy, threshold=10000):
    return 0 if energy < threshold else 1


def strength_binary(strength):
    return 1 if strength > 1 else strength


def load_data(path_to_data, threshold):
    print('Wczytywanie danych...')
    df = pd.read_excel(path_to_data, dtype={'UWAGI': str})
    df.TYP = df.TYP.mask(df.TYP.isin(['O', 'T', 0]), 'OTHER')
    df = df.merge(pd.get_dummies(df.TYP), left_index=True, right_index=True)
    df.drop(['Y'], inplace=True, axis=1)
    df['REJON_ODDZIAL'] = df['REJON'] + df['ODDZIAL']
    amount_big = df[df['ENG'] >= threshold]['REJON_ODDZIAL'].value_counts()
    rej_od = amount_big[amount_big > 10].index
    df = df[df['REJON_ODDZIAL'].isin(rej_od)]
    df.DATA = pd.to_datetime(df.DATA)
    for val in ['Y', 'm', 'd']:
        df[val] = df.DATA.apply(lambda x: int(x.strftime(f'%{val}')))
    df.DATA = df[['Y', 'm', 'd', 'GODZ', 'MIN', 'SEK']].apply(lambda x: datetime.datetime(*x), axis=1)
    df = df.set_index('DATA', drop=True)
    df['SILA'] = df['ENG'].apply(lambda x: add_strength(x, threshold=threshold))
    print('Wczytano dane')
    return df[['ENG', 'SILA', 'OTHER', 'SL', 'W', 'REJON_ODDZIAL']]


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X)[:, :, :-1], np.array(y)[:, :, -1]


def process_data(df, rej_odzzial):
    df_temp = df[df['REJON_ODDZIAL'] == rej_odzzial]
    df_temp.sort_index(inplace=True)
    df_temp = df_temp.resample('8h').sum()
    df_temp['SILA_ILE'] = df_temp['SILA'].copy()
    df_temp['SILA'] = df_temp['SILA'].apply(strength_binary)
    df_temp = df_temp[['ENG', 'OTHER', 'SL', 'W', 'SILA_ILE', 'SILA']]
    return df_temp


def bootstrapping(df_temp, n_steps_in=20, n_steps_out=3):
    print('Rozpoczęto przetwarzanie danych...')
    X, y = split_sequences(df_temp.values, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

    df_for_rf = pd.DataFrame(X).merge(pd.DataFrame(y), left_index=True, right_index=True)

    X = df_for_rf.iloc[:, :-n_steps_out]
    y = df_for_rf.iloc[:, -n_steps_out:]

    y_0 = y.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y_0, test_size=0.33, random_state=42, shuffle=True)
    positive_idx = y_train[y_train == 1].index
    X_pos = X_train.loc[positive_idx, :]

    for i in tqdm.tqdm(range(int(X.shape[1] / n_features))):
        df_samp = X_pos.iloc[:, n_features * i:n_features * (i + 1)].sample()
        for _ in (range(2500)):
            df_samp = df_samp.append(X_pos.iloc[:, n_features * i:n_features * (i + 1)].sample())
        df_samp = df_samp.reset_index(drop=True)
        if i == 0:
            df_samp2 = df_samp.copy()
        else:
            df_samp2 = df_samp2.merge(df_samp, left_index=True, right_index=True)

    X_train = X_train.append(df_samp2)
    y_ones = pd.Series([1] * df_samp2.shape[0])

    y_train = y_train.append(y_ones)

    return X_train, y_train, X_test, y_test


def predict_y(df, rej_oddzial):
    df_temp = process_data(df, rej_odzzial=rej_oddzial)

    X_train, y_train, X_test, y_test = bootstrapping(df_temp)
    rf_0 = RandomForestClassifier(max_depth=5, random_state=123)
    rf_0.fit(X_train, y_train)

    y_pred_class = rf_0.predict(X_test)
    y_pred = rf_0.predict_proba(X_test)

    return y_pred_class, y_pred[:, 1], y_test


def map_values(x):
    d = {0: 'NIE', 1: 'TAK'}
    return d[x]


def get_text(threshold, recall1, recall0, accuracy):
    text = f"""
Wnioski z rozważanego modelu:

Przyjmując, że wybuch uznajemy za duży jeżeli jego siła przekracza {threshold}, przedstawiony model
wykazuje czułość (recall)* dla dużych wybuchów na poziomie {int(recall1 * 100)}%,
a dla małych wybuchów na poziomie {int(recall0 * 100)}%.
Ogólna skuteczność modelu wynosi  {int(accuracy * 100)}%.

*Czułość (recall) to proporcja obserwacji skutecznie przewidzianych jako pochodząca z danej klasy do liczby 
obserwacji pochodzących z tej  klasy.
    """
    return text


def choose_object(df):
    unique = sorted(df.REJON_ODDZIAL.unique())
    n = len(unique)
    d = {i + 1: unique[i] for i in range(n)}
    print('Aby wybrać Rejon i oddział wybierz odpowiedni numer')
    order = list()
    for i in range(n):
        rej, oddzial = unique[i][:2], unique[i][2:]
        order.append((rej, oddzial, i+1))
    rej_od_num = pd.DataFrame(order, columns = ['REJON', 'ODDZIAL', 'NUMER'])
    print(rej_od_num)
    flag = True
    while flag:
        try:
            chosen = int(input('Numer wybranego Regionu i Oddziału: '))
            chosen = d[chosen]
            print(f'Wybrano region {chosen[:2]}, oddział {chosen[2:]}')
            y_pred_class, y_pred, y_test = predict_y(df, chosen)
            flag = False
        except:
            print('Błędnie wprowadzone dane. Spróbuj ponownie.')
    return y_pred_class, y_pred, y_test
