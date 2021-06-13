import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, recall_score
from src.utils import map_values, predict_y, get_text, load_data, choose_object

pd.get_option("display.max_columns")


def main():
    flag = True
    while flag:
        try:
            threshold = input(
                "Wskaż liczbę Joule'i powyżej ilu wstrząs zostanie uznany za mocny (domyślna wartość wynosi 10_000): ")
            threshold = 10_000 if threshold == '' else int(threshold)
            print(f'Przyjęto wartość {threshold}')
            day_time = int(input('Wprowadź liczbę odpowiadającej porze dnia na którą chcesz przewidywać wstrząsy\n0 - 00:00 - 08:00\n1 - 08:00 - 16:00\n2 - 16:00 - 00:00\n'))
            flag = False
        except ValueError:
            print('Wprowadzona wartość jest nieprawidłowa. Spróbuj jeszcze raz.')


    path_to_data = '../data/2016-2020.xlsx'
    df = load_data(path_to_data, threshold)
    y_pred_class, y_pred, y_test = choose_object(df, day_time=day_time)
    print(classification_report(y_test, y_pred_class))

    df = pd.DataFrame({'Czy pojawił się wybuch?': list(map(map_values, y_test)),
                       'Czy przewidywano wybuch?': list(map(map_values, y_pred_class)), 'Prawdop. wyst. wybuchu': y_pred})

    positive = df[df['Czy pojawił się wybuch?'] == 'TAK']
    negative = df[df['Czy pojawił się wybuch?'] == 'NIE']
    print('-'*70)
    input('Wciśnij enter aby kontynuować...')
    print(f'Przykład predykcji, gdy wybuch o energii powyżej {threshold} się pojawił')
    print(positive.sample(10))
    input('Wciśnij enter aby kontynuować...\n\n')
    print(f'Przykład predykcji, gdy wybuch o energii powyżej {threshold} się NIE pojawił')
    print(negative.sample(10))
    input('Wciśnij enter aby kontynuować...\n\n')

    recall1 = recall_score(y_pred=y_pred_class, y_true=y_test)
    recall0 = recall_score(y_pred=y_pred_class, y_true=y_test, pos_label=0)
    accuracy = accuracy_score(y_pred=y_pred_class, y_true=y_test)
    print(get_text(threshold, recall1, recall0, accuracy))

if __name__ == '__main__':
    main()