#This is a complete solution to the loan status prediction problem.
#It first analyzes the training data set. Then it fills up blank values in that data set
#by analyzing patterns in the original data. It uses K-folding cross validation technique with
#5 folds to analyze the generalizability of training data set and generates a predictive model. It also provides
#flexibility in choosing different models among Logistic Regressing, Randomforest classifier, Decision Tree Model etc.
#Finally user can test this model against the test dataset to predict Loan Status based on other qualifiers




#imports
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas import crosstab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import csv
import os

#graph functions
def applicantIncomeHistogram(df):
    df.hist(column="ApplicantIncome", bins=50, figsize=(7, 7), color="blue")
    plt.xlabel("Income")
    plt.ylabel("People")
    return plt.gcf()
def incomeByGenderBoxplot(df):
    df.boxplot(column="ApplicantIncome", by="Gender", figsize=(7,7))
    plt.xticks(rotation=45)
    plt.ylabel("Income")
    return plt.gcf()
def incomeByGenderEducationBoxplot(df):
    df.boxplot(column="ApplicantIncome", by=["Gender","Education"], figsize=(7,9))
    plt.xticks(rotation=45)
    plt.ylabel("Income")
    return plt.gcf()
def loanAmountHistogram(df):
    df.hist(column="LoanAmount", bins=50, color="blue")
    plt.xlabel("Loan Amount")
    plt.ylabel("People")
    return plt.gcf()

def loanAmountHistogram(df):
    df.hist(column="LoanAmount", bins=50, figsize=(7, 7), color="blue")
    plt.xlabel("LoanAmount")
    plt.ylabel("People")
    return plt.gcf()
def loanAmountBoxplot(df):
    df.boxplot(column="LoanAmount")
    plt.xticks(rotation=45)
    plt.ylabel("LoanAmount")
    return plt.gcf()
def loanAmountByGenderBoxplot(df):
    df.boxplot(column="LoanAmount", by="Gender")
    plt.xticks(rotation=45)
    plt.ylabel("LoanAmount")
    return plt.gcf()
def loanAmountByJobBoxplot(df):
    df.boxplot(column="LoanAmount", by="Self_Employed")
    plt.xticks(rotation=45)
    plt.ylabel("LoanAmount")
    return plt.gcf()
def loanAmountByJobEducationBoxplot(df):
    df.boxplot(column="LoanAmount", by=["Self_Employed", "Education"])
    plt.xticks()
    plt.ylabel("LoanAmount")
    return plt.gcf()
def loanAmountLoanTermScatter(df):
    df.plot.scatter(x="LoanAmount",y="Loan_Amount_Term")
    plt.xticks()
    plt.ylabel("Loan_Amount_Term")
    return plt.gcf()
def loanTermBoxplot(df):
    df.boxplot(column="Loan_Amount_Term")
    plt.xticks()
    plt.ylabel("Loan_Amount_Term")
    return plt.gcf()


#text functions
def describeIncome(df):
    numericalIncomeSummary = df['ApplicantIncome'].describe()
    categoricalIncomeSummary= df['Income_Categorization'].value_counts()
    return {'numericalIncomeSummary':numericalIncomeSummary, 'categoricalIncomeSummary':categoricalIncomeSummary}
def describeLoanAmount(df):
    loanAmount = df['LoanAmount'].describe()
    return loanAmount
def creditHistorySummary(df):
    return df['Credit_History'].value_counts(ascending=True)
def loanCreditHistoryPivot(df):  #Probility of getting loan for each Credit History class
    return df.pivot_table(index=['Credit_History'], values=['Loan_Status'], aggfunc=lambda x:x.map({'Y':1, 'N':0}).mean())
def loanMarriedPivot(df):
    return df.pivot_table(index=['Married'], values=['Loan_Status'], aggfunc=lambda x:x.map({'Y':1, 'N':0}).mean())
def loanPropertyPivot(df):
    return df.pivot_table(index=['Property_Area'], values=['Loan_Status'], aggfunc=lambda x:x.map({'Y':1, 'N':0}).mean())
def loanDependentsPivot(df):
    return df.pivot_table(index=['Dependents'], values=['Loan_Status'], aggfunc=lambda x:x.map({'Y':1, 'N':0}).mean())
def loanEducationGenderPivot(df):
    return df.pivot_table(index=['Education'], columns=['Gender'], values=['Loan_Status'], aggfunc=lambda x:x.map({'Y':1, 'N':0}).mean())
def loanEducationPivot(df):
    return df.pivot_table(index=['Education'],  values=['Loan_Status'], aggfunc=lambda x:x.map({'Y':1, 'N':0}).mean())
def loanIncomePivot(df):
    return df.pivot_table(index=['Income_Categorization'], values=['Loan_Status'],
                                 aggfunc=lambda x: x.map({'Y': 1, 'N': 0}).mean())
def loanamountJobPivot(df):
    return df.pivot_table(index=['Self_Employed'], values=['LoanAmount'],
                                 aggfunc=lambda x: x.median())
def loanamountEducationJobPivot(df):
    return df.pivot_table(index='Self_Employed', values='LoanAmount', columns="Education",
                                 aggfunc=lambda x: x.median())
def loanStatusCreditHistoryCrossTab(df):
    return pd.crosstab(index=df['Loan_Status'], columns=df['Credit_History'], normalize="index")   #OR .apply(func=lambda x:x/len(df), axis=0)

def marriedDependentCrossTab(df):
    return pd.crosstab(index=df['Married'], columns=df['Dependents'], normalize="index")   #OR .apply(func=lambda x:x/len(df), axis=0)

def genderMarriedCrossTab(df):
    return pd.crosstab(index=df['Gender'], columns=df['Married'], normalize="index")   #OR .apply(func=lambda x:x/len(df), axis=0)
def genderEducationPivot(df):
    return df.pivot_table(index=['Gender'], values=['Education'],
                                 aggfunc=lambda x: x.map({'Graduate': 1, 'Not Graduate': 0}).mean())
def marriedEducationGenderPivot(df):
    return df.pivot_table(index=['Married'], columns=['Education'], values=['Gender'],
                                 aggfunc=lambda x: x.map({'Male': 1, 'Female': 0}).mean())

def selfEmployedCount(df):
    return df['Self_Employed'].value_counts()
def genderCount(df):
    return df['Gender'].value_counts(normalize="columns")
def marriedCount(df):
    return df['Married'].value_counts(normalize="columns")
def loanTermCount(df):
    return df['Loan_Amount_Term'].value_counts()
def dependentsCount(df):
    return df['Dependents'].value_counts()
#datamunging
def countMissingValues(df):
    return df.apply(func=lambda x: sum(x.isnull()) ,axis=0)
def fillCategorizeIncome(df):
        df['Income_Categorization']=df.apply(func=lambda x:classifyIncome(int(x['ApplicantIncome']), int(x['CoapplicantIncome'])), axis=1)
def fillGender(df):
        df['Gender'].fillna(df[df['Gender'].isnull()].apply(func=lambda x:guessGender(df, x), axis=1), inplace=True)
def fillMarried(df):
        df['Married'].fillna(df[df['Married'].isnull()].apply(func=lambda x:guessMarriageStatus(df, x), axis=1), inplace=True)
def fillDependents(df):
        df["Dependents"].fillna(df[df["Dependents"].isnull()].apply(func=lambda x: guessDependents(df, x), axis=1), inplace=True)
def fillSelfEmployed(df):
        df['Self_Employed'].fillna(value="No", inplace=True)
def fillLoanTerm(df):
        df["Loan_Amount_Term"].fillna(value=df["Loan_Amount_Term"].median(), inplace=True)
def fillCreditHistory(df):
        df["Credit_History"].fillna(df[df["Credit_History"].isnull()].apply(func=lambda x:guessCreditHistory(df, x), axis=1), inplace=True)
def fillLoanAmount(df):
        table=loanamountEducationJobPivot(df)
        df["LoanAmount"].fillna(df[df["LoanAmount"].isnull()].apply(func=lambda x: table.loc[x["Self_Employed"],x["Education"]], axis=1), inplace=True)
#helper functions
def classifyIncome(applicantIncome, coapplicantIncome):
    lowceil=1500
    medceil=4000
    highceil=8000
    if(applicantIncome+coapplicantIncome>highceil):
        return "upperclass"
    elif(applicantIncome+coapplicantIncome>medceil):
        return "uppermiddleclass"
    elif(applicantIncome+coapplicantIncome> lowceil):
         return "middleclass"
    else:
        return "lowclass"
def guessGender(df, x):
    table=genderCount(df)
    rand=np.random.uniform(low=0.0, high=1.0, size=1)
    if(rand<table.loc['Male',] and rand>=0.0):
        return "Male"
    else:
        return "Female"

def guessMarriageStatus(df, x):
    table=genderMarriedCrossTab(df)
    notMarriedWomenThresh= table.loc["Female","No"]
    notMarriedMenThresh=table.loc["Male", "No"]
    print(notMarriedWomenThresh)
    print(notMarriedMenThresh)
    rand = np.random.uniform(low=0.0, high=1.0, size=1)
    if((x["Gender"]=="Male" and rand<notMarriedMenThresh) or (x["Gender"]=="Female" and rand < notMarriedWomenThresh)):
        print(x["Gender"] + "No")
        print(("Rand = {}").format(rand))
        return("No")
    else:
        print(x["Gender"]+" Yes")
        print(("Rand = {}").format(rand))
        return("Yes")

def guessCreditHistory(df, x):
    #table = loanStatusCreditHistoryCrossTab(df)
    #noLoanNoCreditThresh = table.loc["N", 0]
    #yesLoanYesCreditThresh = table.loc["Y",1]
    #print(noLoanNoCreditThresh)
    #print(yesLoanYesCreditThresh)
    rand = np.random.uniform(low=0.0, high=1.0, size=1)
    if (rand < 0.5):
        return (0)
    else:
        return (1)

def guessDependents(df, x):
    table = marriedDependentCrossTab(df)
    #print(x["Dependents"].dtype)
    notMarriedBins=[0, table.loc["No", "0"], table.loc["No", "0"]+table.loc["No", "1"],table.loc["No", "0"]+table.loc["No", "1"]+table.loc["No", "2"],table.loc["No", "0"]+table.loc["No", "1"]+table.loc["No", "2"]+table.loc["No", "3+"]]
    yesMarriedBins=[0, table.loc["Yes", "0"], table.loc["Yes", "0"]+table.loc["Yes", "1"],table.loc["Yes", "0"]+table.loc["Yes", "1"]+table.loc["Yes", "2"],table.loc["Yes", "0"]+table.loc["Yes", "1"]+table.loc["Yes", "2"]+table.loc["Yes", "3+"]]
    print("married bins: ")
    print(yesMarriedBins)
    print("not married bins: ")
    print(notMarriedBins)
    labels=[0,1,2,3]
    rand=np.random.uniform(low=0.0, high=1.0, size=1)
    if(x["Married"] == "No"):
        resultBin=pd.cut(rand, bins=notMarriedBins, right=True, labels=labels)
        print("Married : "+ x["Married"]+ "Rand : {}".format(rand)+ "Resultbin : {}".format(resultBin))
        return {
            0:"0",
            1:"1",
            2:"2",
            3:"3+"
        }[resultBin[0]]
    elif (x["Married"] == "Yes"):
        resultBin = pd.cut(rand, bins=yesMarriedBins, right=True, labels=labels)
        print("Married : " + x["Married"] + " Rand : {}".format(rand) + "Resultbin : {}".format(resultBin))
        return {
            0:"0",
            1:"1",
            2:"2",
            3:"3+"
        }[resultBin[0]]

    else:
        return "0"
def doLogisticTransformation(df):
    df["LoanAmount_Log"]=np.log(df["LoanAmount"])
    df["TotalIncome_Log"]=np.log(df["ApplicantIncome"]+df["CoapplicantIncome"])

def convertToNumeric(df):
    list=["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    labelEncoder=LabelEncoder()
    for i in list:
        df[i]=labelEncoder.fit_transform(df[i])
    return df

#data modelling and predictions
def testClassificationModels(df):
    #Logistic regression model
    model=LogisticRegression()
    outcomeVariable="Loan_Status"
    predictorVariables=["Credit_History", "Married", "Gender", "Education"]
    classificationModel(model=model, data=df, predictors=predictorVariables, outcome=outcomeVariable)
    #Decision tree model
    model = DecisionTreeClassifier()
    #outcomeVariable = "Loan_Status"
    #predictorVariables = ["Credit_History", "Married", "Gender", "Education", "LoanAmount_Log"]
    classificationModel(model=model, data=df, predictors=predictorVariables, outcome=outcomeVariable)
    #Random forest classifier
    model = RandomForestClassifier()
    #outcomeVariable = "Loan_Status"
    #predictorVariables = ["Gender", "Married", "Dependents", "Education",
     #  "Self_Employed", "Loan_Amount_Term", "Credit_History", "Property_Area",
      #  "LoanAmount_Log","TotalIncome_Log"]
    classificationModel(model=model, data=convertToNumeric(df), predictors=predictorVariables, outcome=outcomeVariable)
def classificationModel(model, data, predictors, outcome):
    #Accuracy
    print(model.__str__())
    model.fit(data[predictors], data[outcome])
    predictions=model.predict(data[predictors])
    #print(predictions)
    accuracy=metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy : {:.3%} ".format(accuracy))
    #KFOLD CrossValidation
    kf=KFold(data.shape[0], n_folds=5)
    error=[]
    for train, test in kf:
        train_predictors=data[predictors].iloc[train,:]
        train_target=data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    print("Cross Validation Score : {:.3%} \n".format(np.mean(error)))

    model.fit(data[predictors], data[outcome])

def logRegToPredictLoanStatus():
    testdatacopy=convertToNumeric(testdatanum)   #numerical copy of the dataframe
    traindatacopy=convertToNumeric(traindatanum)

    model = LogisticRegression()
    outcomeVariable = "Loan_Status"
    predictorVariables = ["Credit_History", "Married", "Gender", "Education"]
    model.fit(traindatacopy[predictorVariables], traindatacopy[outcomeVariable])
    testdata[outcomeVariable] = model.predict(testdatacopy[predictorVariables])

    
#output functions

def exportGraphs(df, filename):
    pdf=PdfPages(filename=filename)
    pdf.savefig(figure=applicantIncomeHistogram(df))
    pdf.savefig(figure=incomeByGenderBoxplot(df))
    pdf.savefig(figure=incomeByGenderEducationBoxplot(df))
    pdf.savefig(figure=loanAmountHistogram(df))
    pdf.savefig(figure=loanAmountBoxplot(df))
    pdf.savefig(figure=loanAmountByGenderBoxplot(df))
    pdf.savefig(figure=loanAmountByJobBoxplot(df))
    pdf.savefig(figure=loanAmountByJobEducationBoxplot(df))
    pdf.savefig(figure=loanAmountLoanTermScatter(df))
    pdf.savefig(figure=loanTermBoxplot(df))
    pdf.close()
def exportText(df, filename):
    outputfile=open(file=filename, mode="w")
    incomeSummary=describeIncome(df)
    outputfile.write(Series(incomeSummary['numericalIncomeSummary']).to_string(name=True)+"\n\n"+Series(incomeSummary['categoricalIncomeSummary']).to_string(name=True)+"\n\n")
    outputfile.write(Series(describeLoanAmount(df)).to_string(name=True) + "\n\n")
    outputfile.write(Series(creditHistorySummary(df)).to_string(name=True)+"\n\n")
    outputfile.write(Series(selfEmployedCount(df)).to_string(name=True) + "\n\n")
    outputfile.write(Series(genderCount(df)).to_string(name=True) + "\n\n")
    outputfile.write(Series(marriedCount(df)).to_string(name=True) + "\n\n")
    outputfile.write(Series(loanTermCount(df)).to_string(name=True) + "\n\n")
    outputfile.write(Series(dependentsCount(df)).to_string(name=True) + "\n\n")
    outputfile.write(DataFrame(loanCreditHistoryPivot(df)).to_string()+"\n\n")
    outputfile.write(DataFrame(loanMarriedPivot(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(loanPropertyPivot(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(loanDependentsPivot(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(loanEducationPivot(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(loanEducationGenderPivot(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(loanIncomePivot(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(loanamountJobPivot(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(genderMarriedCrossTab(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(genderEducationPivot(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(marriedEducationGenderPivot(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(loanamountEducationJobPivot(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(loanStatusCreditHistoryCrossTab(df)).to_string() + "\n\n")
    outputfile.write(DataFrame(marriedDependentCrossTab(df)).to_string() + "\n\n")
    outputfile.write(Series(countMissingValues(df)).to_string(name=True) + "\n\n")
    outputfile.write(Series(df.dtypes).to_string(name=True)+"\n\n")
    #outputfile.write(Series(df["Property_Area"]).to_string(name=True) + "\n\n")
    outputfile.close()

def fillMissingValues(df):  #comment out the ones that are already filled otherwise it will give an error
    fillGender(df)
    #fillMarried(df)                 //comment this out for test data since test data has all values for marriage status
    fillDependents(df)
    fillSelfEmployed(df)
    fillCategorizeIncome(df)
    fillLoanAmount(df)
    fillLoanTerm(df)
    fillCreditHistory(df)

#uncomment the ones that you need
traindata= pd.read_csv("data/train.csv")
testdata= pd.read_csv("data/test.csv")
traindatanum= pd.read_csv("data/train.csv")   #copy of original data with categorical variables represented as numbers
testdatanum= pd.read_csv("data/test.csv")
#fillMissingValues(testdata)
#fillMissingValues(traindata)
#doLogisticTransformation(testdata)
#doLogisticTransformation(traindata)


#testClassificationModels(traindata)
#testClassificationModels(traindata)
#logRegToPredictLoanStatus()
exportText(traindata,filename="traindatasummary.txt")
exportGraphs(traindata,filename="graphsTrainData.pdf")
exportText(testdata,filename="testdatasummary.txt")
exportGraphs(testdata,filename="graphsTestData.pdf")
#testdata.to_csv(path_or_buf="data/test.csv", index=False)
#traindata.to_csv(path_or_buf="data/test.csv", index=False)

