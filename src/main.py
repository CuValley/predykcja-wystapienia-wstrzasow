from src.utils import predict_y

if __name__ == '__main__':
    y_pred_class, y_pred, y_test = predict_y(threshold=10E3)
    print(y_pred)