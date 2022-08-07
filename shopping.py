import csv
import sys
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open("d:\CS50 AI\Shopping\shopping\shopping.csv") as f:
        reader = csv.reader(f)
        next(reader)

        data = []
        data1 = []
        #flag=0 #to check if data is being printed in the correct format
        for row in reader:
            if(row[10]=="Jan"):
                ans10=0
            elif(row[10]=="Feb"):
                ans10=1
            elif(row[10]=="Mar"):
                ans10=2
            elif(row[10]=="Apr"):
                ans10=3
            elif(row[10]=="May"):
                ans10=4
            elif(row[10]=="June"):
                ans10=5
            elif(row[10]=="Jul"):
                ans10=6
            elif(row[10]=="Aug"):
                ans10=7
            elif(row[10]=="Sep"):
                ans10=8
            elif(row[10]=="Oct"):
                ans10=9
            elif(row[10]=="Nov"):
                ans10=10
            else:
                ans10=11
            if(row[15]=="Returning_Visitor"):
                ans15=1
            else:
                ans15=0
            if(row[16]=="FALSE"):
                ans16=0
            else:
                ans16=1
            data1.append(
                [int(row[0]),
                float(row[1]),
                int(row[2]),
                float(row[3]),
                int(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
                float(row[8]),
                float(row[9]),
                int(ans10),#hmm
                int(row[11]),
                int(row[12]),
                int(row[13]),
                int(row[14]),
                int(ans15),#hmm
                int(ans16)])#hmm
            data.append( 1 if row[17] == "TRUE" else 0)
            #if flag<100:
                #print(data1[flag])
                #flag+=1

            
    return (data1,data)
    raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
# Fit model
    model.fit(evidence,labels)

# Make predictions on the testing set
    #predictions = model.predict(X_testing)
    return model
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    t_p=0
    t_n=0
    f_p=0
    f_n=0
    for i in range(len(labels)):
        if(labels[i]==1 and predictions[i]==1):
            t_p+=1
        if(labels[i]==1 and predictions[i]==0):
            f_n+=1
        if(labels[i]==0 and predictions[i]==1):
            f_p+=1
        if(labels[i]==0 and predictions[i]==0):
            t_n+=1
          
     
        
    a1=float(t_p/(t_p+f_n))
    a2=float(t_n/(t_n+f_p))
    return (a1,a2)
    raise NotImplementedError


if __name__ == "__main__":
    main()
