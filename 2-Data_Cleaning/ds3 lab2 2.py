import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

data = pd.read_csv("landslide_data3_miss.csv", sep=",")
or_data =  pd.read_csv("landslide_data3_original.csv", sep=",")


# making dataframe of the both data
df= pd.DataFrame(data)
or_df = pd.DataFrame(or_data)

length = len(df["stationid"])
attributes = list(df.columns)

# 1
# count the no. of NaN values in each attributes
counts =df.isna().sum()
print(counts)
# Plot the graph
plt.bar(attributes, counts , width= 0.75, label = "number of missing values")
plt.legend()
plt.show()

# 2 a.
#  dropping the rows having NaN in the stationID attribute.
noSI1 = df.dropna(subset=["stationid"])
print()
# finding no. of tuples deleted
diff = len(df["stationid"])- len(noSI1["stationid"])
print(f"the total number of tuples deleted = {diff}")

# 2b
#  dropping the rows having NaN more tha 2
noSI = noSI1.dropna(axis = 0 , thresh =3)
# count total tuples deleted after 2 operations
diff = len(df["stationid"])- len(noSI1["stationid"])
print(f"the total number of tuples deleted = {diff}")


# 3
# find missing values in each attributes and also the sum of all missing values
counts =noSI.isna().sum()

print(counts)
print(sum(counts))


# 4 a
# Replace the missing values by mean of their respective attribute.
attributes = list(noSI.columns)

# forming a new dataframe new and filling it with replaced values
new = pd.DataFrame()
for i in attributes:
    if(i != "dates" and i != "stationid" ):
        new[i]=noSI[i].fillna(noSI[i].mean() )
print(new)


# a. 1 Computing mean, median, mode and standard deviation for each attributes and compare
# this is the case of replacing Nan values with mean of the attribute

for name in attributes:
    if(name =="stationid" or name=="dates"):
        continue
    mean1 = new[name].mean()
    mean2 = or_df[name].mean()
    print(f"mean of attribute {name} : edited- {mean1} , original - {mean2}")
    med1 = np.median(new[name])
    med2 = np.median(or_df[name])
    print(f"median of attribute {name} : edited- {med1} , original - {med2}")
    mode1= statistics.mode(new[name])
    mode2= statistics.mode(or_df[name])

    print(f"mode of attribute {name}  : edited- {mode1} , original - {mode2}")
    std1 = statistics.stdev(new[name])
    std2 = statistics.stdev(or_df[name])

    print(f"Standard Deviation of attribute {name} : edited- {std1} , original - {std2}")

# ii. Calculate the root mean square error (RMSE) between the original and replaced values for each attribute.
# places  which had nan values
noSIbool = noSI.isna()
# listb of indexes
rows = list(noSI.index)
# form an RMSE list to store RMSE of all the attributes
RMSE=[]
# filling rmse of each attributes
realatt = attributes[2:]
for name in attributes:
    misscount = 0
    rms =0
    for i in rows:
        if (name == "stationid" or name == "dates"):
            continue

        else:
            if noSIbool[name][i]==True:
                misscount +=1
                rms += (or_df[name][i]-new[name][i])**2
    RMSE.append(round((rms**0.5)/(max(misscount,1)**0.5),2))

RMSEbyMean = pd.Series(RMSE[2:] , index=realatt)

# plotting graph
plt.bar( realatt,RMSE[2:], label = "RMSE")
plt.title("Replacing missing values with mean .")
plt.xlabel("attributes")
plt.show()

print("RMSEbyMean")
print(RMSEbyMean)

# Replace the missing values in each attribute using linear interpolation technique. Usedf.interpolate() with suitable arguments.
inter = noSI.interpolate()
print(inter)
# 1
for name in attributes:
    if(name =="stationid" or name=="dates"):
        continue
    mean1 = inter[name].mean()
    mean2 = or_df[name].mean()
    print(f"mean of attribute {name} : edited- {mean1} , original - {mean2}")
    med1 = np.median(inter[name])
    med2 = np.median(or_df[name])
    print(f"median of attribute {name} : edited- {med1} , original - {med2}")
    mode1= statistics.mode(inter[name])
    mode2= statistics.mode(or_df[name])

    print(f"mode of attribute {name}  : edited- {mode1} , original - {mode2}")
    std1 = statistics.stdev(inter[name])
    std2 = statistics.stdev(or_df[name])

    print(f"Standard Deviation of attribute {name} : edited- {std1} , original - {std2}")

# ii. Calculate the root mean square error (RMSE) between the original and replaced values for each attribute.
# forming RMSE
RMSE=[]
for name in attributes:
    misscount = 0
    rms =0
    for i in rows:
        if (name == "stationid" or name == "dates"):
            continue

        if noSIbool[name][i]==True:
            misscount +=1
            rms += (or_df[name][i]-inter[name][i])**2
    RMSE.append(round((rms**0.5)/((max(misscount,1))**0.5),2))
RMSEbyInterpolation = pd.Series(RMSE[2:] , index=attributes[2:])

#plotting graph
plt.bar( realatt,RMSE[2:], label = "RMSE")
plt.title("Replacing missing values with interpolation.")
plt.xlabel("attributes")

plt.show()

print("RMSEbyInterpolation")
print(RMSEbyInterpolation)


# 5
# Box plotting rain and temperature

plt.boxplot(inter["temperature"])
plt.title("attribute = temperature")
plt.show()

plt.boxplot(inter["rain"])
plt.title("attribute = rain")
plt.show()

# finding IQR and then Outliers

rt =[ "temperature", "rain"]
des = inter.describe()
print(des)

# list of outliers for each attribute- rain and temperature
for name in rt:
    outliers=[]
    outliersrow =[]
    med = np.median(inter[name])
    IQR = des[name]["75%"]-des[name]["25%"]
    print(IQR)
    # lower and upper limit
    ll = des[name]["25%"] - 1.5*IQR
    ul = des[name]["75%"] + 1.5*IQR
    # checking outlier condition and replacing it with the median
    for i in rows:
        if(ll>inter[name][i] or ul<inter[name][i] ):
            outliers.append(inter[name][i])
            outliersrow.append(i)
            inter.loc[i, name]=med
    print(len(outliersrow))
    print(outliersrow)
    print("outliers of "+ name)
    print(outliers)


# boxplot after replacement
plt.boxplot(inter["temperature"])
plt.title("attribute = temperature")
plt.show()

plt.boxplot(inter["rain"])
plt.title("attribute = rain")
plt.show()

des = inter.describe()
for name in rt:
    outliers=[]
    outliersrow =[]
    med = np.median(inter[name])
    IQR = des[name]["75%"]-des[name]["25%"]
    print(IQR)
    # lower and upper limit
    ll = des[name]["25%"] - 1.5*IQR
    ul = des[name]["75%"] + 1.5*IQR
    # checking outlier condition and replacing it with the median
    for i in rows:
        if(ll>inter[name][i] or ul<inter[name][i] ):
            outliers.append(inter[name][i])
            outliersrow.append(i)
    print(len(outliersrow))
    print(outliersrow)
    print("outliers of "+ name)
    print(outliers)



