import sys

def main():
    pred_path = sys.argv[1]
    label_path = sys.argv[2]

    with open(pred_path, encoding='utf-8') as f:
        preds = [l.strip() for l in f.readlines()]
    with open(label_path, encoding='utf-8') as f:
        labels = [l.strip() for l in f.readlines()]
    assert len(preds) == len(labels), "Length of predictions ({}) and labels ({}) are not the same"\
        .format(len(preds), len(labels))
    
    correct = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            correct += 1
    print('Accuracy: {:.2f}%'.format((100.0 * correct) / len(labels)))

if __name__ == "__main__":
    main()