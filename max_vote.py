import csv

predictions = {}


def max_vote(filenames):
    # Takes a list of filenames and creates a new prediction that always contains the value the majority of the models voted for.
    for file in filenames:
        with open(file, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                id, pred = int(line.split(",")[0]), int(line.split(",")[1])
                if id not in predictions:
                    predictions[id] = 0
                predictions[id] += pred
    with open("majority_vote.csv", "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for id in predictions:
            pred = -1
            if predictions[id] > 0:
                pred = 1
            writer.writerow({"Id": int(id), "Prediction": int(pred)})
