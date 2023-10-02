from sklearn.preprocessing import MinMaxScaler


def preprocessing(x):
    model = MinMaxScaler()
    return model.fit_transform(x.astype("float32"))
