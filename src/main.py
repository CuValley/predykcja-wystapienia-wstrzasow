# from src.utils import *

flag = True
while flag:
    try:
        threshold = int(input("Wskaż liczbę Joule'i powyżej ilu wstrząs zostanie uznany za mocny (domyślna wartość wynosi 10_000): "))
        flag = False
    except ValueError:
        print('Wprowadzona wartość jest nieprawidłowa. Spróbuj jeszcze raz.')