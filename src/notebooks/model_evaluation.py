import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, classification_report, roc_auc_score
from IPython.display import display


def print_metrics(model, X_train, y_train, X_test, y_test):
    pred = model.predict(X_test)

    report = classification_report(y_test, pred, output_dict=True)

    test_acc = report.pop("accuracy")
    train_acc = model.score(X_train, y_train)

    report_df = pd.DataFrame(report)

    display(report_df.style.set_caption('Classification Report'))

    print(f"Train Accuracy: {train_acc:.4f}\t\tTest Accuracy: {test_acc:.4f}")
    roc_graph = RocCurveDisplay.from_estimator(model, X_test, y_test)
    roc_graph.ax_.set_title(f'AUC: {roc_graph.roc_auc}')
    roc_graph.ax_.get_legend().remove()
    roc_graph.figure_.set_size_inches(8, 6)


def print_metrics_nn(model, X_train, y_train, X_test, y_test):
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    y_pred = model.predict(X_test)
    y_pred = y_pred > 0.5

    report = classification_report(y_test, y_pred, output_dict=True)
    report.pop("accuracy")
    report_df = pd.DataFrame(report)

    display(report_df.style.set_caption('Classification Report'))

    print(f"Train Accuracy: {train_acc:.4f}\t\tTest Accuracy: {test_acc:.4f}")

    auc_score = roc_auc_score(y_test, model.predict(X_test).ravel())
    print(f"AUC Score: {auc_score}")


def nn_compare_train_val(nn_history):
    num_epochs = nn_history.params["epochs"]
    history = nn_history.history

    loss_data = {
        "epochs": range(1, num_epochs+1),
        "train_loss": history["loss"],
        "val_loss": history["val_loss"]
    }
    loss_df = pd.DataFrame(loss_data)

    acc_data = {
        "epochs": range(1, num_epochs+1),
        "train_accuracy": history["accuracy"],
        "val_accuracy": history["val_accuracy"]
    }
    acc_df = pd.DataFrame(acc_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    sns.lineplot(data=pd.melt(loss_df, "epochs"), x='epochs',
                 y='value', hue='variable', ax=ax1)
    ax1.set_ylabel("Loss")

    sns.lineplot(data=pd.melt(acc_df, "epochs"), x='epochs',
                 y='value', hue='variable', ax=ax2)
    ax2.set_ylabel("Accuracy")

    plt.tight_layout()
