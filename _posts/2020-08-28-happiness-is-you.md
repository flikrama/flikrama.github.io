---
title: The happiness project
categories:
- Linear Regression
- Ridge 
- Lasso
- Elastic Net Regression
- Data Cleaning
- Data Manipulation

---
## Happiness is you

Would you rather be rich or happy? I'd rather be both...and healthy too! 
 
Many people, organizations and even governments have started looking at happiness as a metric to measure success besides economic indicators.
Being happy is a simple yet extremely profound feeling. It is also hard to define. Happiness Index may seem trivial, but it points to the gaps in the government development policies in each country and one may potentially view the measure as people’s perception of how their governments perform.  The insights from its analysis also helps in pointing out the importance of development in several aspects instead of just narrowly in economic or health indicators.

This work is focused on datasets (from 2015 to 2019) from the World Happiness Report Landmark Survey. Its objective is predicting the happiness score of a country based on independent features and identifying the key variables and potentially their interactions.
This project was part of the UH SPE Machine Learning Bootcamp and collaborators were myself, Celine Cherian, Miguel Mendoza and Pratik Ghatake. Here goes below:

* [Libraries ](#libraries)
* [Data loading, cleaning and visualizations](#data-loading-cleaning-and-visualizations)
* [Descriptive Analytics](#Descriptive-Analytics)
* [Regression Analysis](#regression-analysis)
    * [Ridge Regression](#Ridge-Regression)
    * [Lasso Regression](#Lasso-Regression)
    * [Elastic Net Regression](#Elastic-Net-Regression)
* [Summary](#Summary)



## Libraries <a id='libraries'></a>



```python
# Data Analysis and Manipulation
import numpy as np
import pandas as pd
from numpy import NaN as NA

# Plotting and Visualization
import matplotlib.pyplot as plt
import seaborn as sb
from bubbly.bubbly import bubbleplot
import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot

# Analysis 
import sklearn.linear_model as lm
from sklearn.feature_selection import rfe
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.metrics import mean_squared_error, r2_score
```

### Data loading, cleaning and visualizations <a id='data-loading-cleaning-and-visualizations'></a>


The United Nations publishes the World Happiness Report every year but years 2015 to 2019 are selected for this analysis. The rankings of the happiness report are based on a Cantril ladder survey. Nationally representative samples of respondents are asked to think of a ladder, with the best possible life for them being a 10, and the worst possible life being a 0. They are then asked to rate their own current lives on that 0 to 10 scale. The report correlates the results with various life factors. There are different fields involved in the dataset including economics, psychology, national statistical figures etc. which are measured on different scales and are used to effectively assess the happiness score of the country. There are a total 782 observations of different countries for the span of 5 years in total with 9 different variables. 



```python
df2015= pd.read_csv('2015.csv') 
df2016= pd.read_csv('2016.csv')
df2017= pd.read_csv('2017.csv')
df2018= pd.read_csv('2018.csv')
df2019= pd.read_csv('2019.csv')
```


```python
yearsData = ['2015','2016','2017','2018','2019']
dfs = [df2015.shape,df2016.shape,df2017.shape,df2018.shape,df2019.shape]

for i in range(len(dfs)):
  print("Current dimensions for {0} are {1}".format(yearsData[i],dfs[i])) 
```

    Current dimensions for 2015 are (158, 12)
    Current dimensions for 2016 are (157, 13)
    Current dimensions for 2017 are (155, 12)
    Current dimensions for 2018 are (156, 9)
    Current dimensions for 2019 are (156, 9)
    


```python
df2015.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Region</th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Standard Error</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Trust (Government Corruption)</th>
      <th>Generosity</th>
      <th>Dystopia Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>7.587</td>
      <td>0.03411</td>
      <td>1.39651</td>
      <td>1.34951</td>
      <td>0.94143</td>
      <td>0.66557</td>
      <td>0.41978</td>
      <td>0.29678</td>
      <td>2.51738</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>2</td>
      <td>7.561</td>
      <td>0.04884</td>
      <td>1.30232</td>
      <td>1.40223</td>
      <td>0.94784</td>
      <td>0.62877</td>
      <td>0.14145</td>
      <td>0.43630</td>
      <td>2.70201</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>3</td>
      <td>7.527</td>
      <td>0.03328</td>
      <td>1.32548</td>
      <td>1.36058</td>
      <td>0.87464</td>
      <td>0.64938</td>
      <td>0.48357</td>
      <td>0.34139</td>
      <td>2.49204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>4</td>
      <td>7.522</td>
      <td>0.03880</td>
      <td>1.45900</td>
      <td>1.33095</td>
      <td>0.88521</td>
      <td>0.66973</td>
      <td>0.36503</td>
      <td>0.34699</td>
      <td>2.46531</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>North America</td>
      <td>5</td>
      <td>7.427</td>
      <td>0.03553</td>
      <td>1.32629</td>
      <td>1.32261</td>
      <td>0.90563</td>
      <td>0.63297</td>
      <td>0.32957</td>
      <td>0.45811</td>
      <td>2.45176</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2016.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Region</th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Lower Confidence Interval</th>
      <th>Upper Confidence Interval</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Trust (Government Corruption)</th>
      <th>Generosity</th>
      <th>Dystopia Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>7.526</td>
      <td>7.460</td>
      <td>7.592</td>
      <td>1.44178</td>
      <td>1.16374</td>
      <td>0.79504</td>
      <td>0.57941</td>
      <td>0.44453</td>
      <td>0.36171</td>
      <td>2.73939</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>2</td>
      <td>7.509</td>
      <td>7.428</td>
      <td>7.590</td>
      <td>1.52733</td>
      <td>1.14524</td>
      <td>0.86303</td>
      <td>0.58557</td>
      <td>0.41203</td>
      <td>0.28083</td>
      <td>2.69463</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>3</td>
      <td>7.501</td>
      <td>7.333</td>
      <td>7.669</td>
      <td>1.42666</td>
      <td>1.18326</td>
      <td>0.86733</td>
      <td>0.56624</td>
      <td>0.14975</td>
      <td>0.47678</td>
      <td>2.83137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>4</td>
      <td>7.498</td>
      <td>7.421</td>
      <td>7.575</td>
      <td>1.57744</td>
      <td>1.12690</td>
      <td>0.79579</td>
      <td>0.59609</td>
      <td>0.35776</td>
      <td>0.37895</td>
      <td>2.66465</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>5</td>
      <td>7.413</td>
      <td>7.351</td>
      <td>7.475</td>
      <td>1.40598</td>
      <td>1.13464</td>
      <td>0.81091</td>
      <td>0.57104</td>
      <td>0.41004</td>
      <td>0.25492</td>
      <td>2.82596</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2017.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Happiness.Rank</th>
      <th>Happiness.Score</th>
      <th>Whisker.high</th>
      <th>Whisker.low</th>
      <th>Economy..GDP.per.Capita.</th>
      <th>Family</th>
      <th>Health..Life.Expectancy.</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust..Government.Corruption.</th>
      <th>Dystopia.Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Norway</td>
      <td>1</td>
      <td>7.537</td>
      <td>7.594445</td>
      <td>7.479556</td>
      <td>1.616463</td>
      <td>1.533524</td>
      <td>0.796667</td>
      <td>0.635423</td>
      <td>0.362012</td>
      <td>0.315964</td>
      <td>2.277027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2</td>
      <td>7.522</td>
      <td>7.581728</td>
      <td>7.462272</td>
      <td>1.482383</td>
      <td>1.551122</td>
      <td>0.792566</td>
      <td>0.626007</td>
      <td>0.355280</td>
      <td>0.400770</td>
      <td>2.313707</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>3</td>
      <td>7.504</td>
      <td>7.622030</td>
      <td>7.385970</td>
      <td>1.480633</td>
      <td>1.610574</td>
      <td>0.833552</td>
      <td>0.627163</td>
      <td>0.475540</td>
      <td>0.153527</td>
      <td>2.322715</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Switzerland</td>
      <td>4</td>
      <td>7.494</td>
      <td>7.561772</td>
      <td>7.426227</td>
      <td>1.564980</td>
      <td>1.516912</td>
      <td>0.858131</td>
      <td>0.620071</td>
      <td>0.290549</td>
      <td>0.367007</td>
      <td>2.276716</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Finland</td>
      <td>5</td>
      <td>7.469</td>
      <td>7.527542</td>
      <td>7.410458</td>
      <td>1.443572</td>
      <td>1.540247</td>
      <td>0.809158</td>
      <td>0.617951</td>
      <td>0.245483</td>
      <td>0.382612</td>
      <td>2.430182</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2018.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.632</td>
      <td>1.305</td>
      <td>1.592</td>
      <td>0.874</td>
      <td>0.681</td>
      <td>0.202</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Norway</td>
      <td>7.594</td>
      <td>1.456</td>
      <td>1.582</td>
      <td>0.861</td>
      <td>0.686</td>
      <td>0.286</td>
      <td>0.340</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Denmark</td>
      <td>7.555</td>
      <td>1.351</td>
      <td>1.590</td>
      <td>0.868</td>
      <td>0.683</td>
      <td>0.284</td>
      <td>0.408</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.495</td>
      <td>1.343</td>
      <td>1.644</td>
      <td>0.914</td>
      <td>0.677</td>
      <td>0.353</td>
      <td>0.138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Switzerland</td>
      <td>7.487</td>
      <td>1.420</td>
      <td>1.549</td>
      <td>0.927</td>
      <td>0.660</td>
      <td>0.256</td>
      <td>0.357</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2019.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Denmark</td>
      <td>7.600</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
      <td>0.252</td>
      <td>0.410</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Norway</td>
      <td>7.554</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
      <td>0.271</td>
      <td>0.341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.494</td>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
      <td>0.354</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Netherlands</td>
      <td>7.488</td>
      <td>1.396</td>
      <td>1.522</td>
      <td>0.999</td>
      <td>0.557</td>
      <td>0.322</td>
      <td>0.298</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2015.drop(['Region','Standard Error','Dystopia Residual'],axis=1, inplace = True) 
print('columns are dropped!')

df2015.rename(columns = {'Trust (Government Corruption)':'Perceptions of corruption','Country':'Country or Region',
                         'Family':'Social Support','Freedom':'Freedom to make life choices'},inplace = True)

# switch columns "Perceptions of Corruption" and "Generosity"
df2015.columns
df2015new = df2015[['Country or Region','Happiness Rank', 'Happiness Score',

       'Economy (GDP per Capita)', 'Social Support',
       'Health (Life Expectancy)', 'Freedom to make life choices','Generosity','Perceptions of corruption']]


# drop columns
df2016.drop(['Region','Lower Confidence Interval','Upper Confidence Interval','Dystopia Residual'],axis=1, inplace = True) 
print('columns are dropped!')

# rename columns
df2016.rename(columns = {'Trust (Government Corruption)':'Perceptions of corruption','Country':'Country or Region',
                         'Family':'Social Support','Freedom':'Freedom to make life choices'},inplace = True)

# switch columns "Perceptions of Corruption" and "Generosity" 
df2016.columns
df2016new = df2016[['Country or Region', 'Happiness Rank', 'Happiness Score',
       'Economy (GDP per Capita)', 'Social Support',
       'Health (Life Expectancy)', 'Freedom to make life choices','Generosity','Perceptions of corruption']]


# drop columns
df2017.drop(['Whisker.high','Whisker.low','Dystopia.Residual'],axis=1, inplace = True) 
print('columns are dropped!')

# rename columns
df2017.rename(columns = {'Trust..Government.Corruption.':'Perceptions of corruption','Country':'Country or Region',
                         'Family':'Social Support','Freedom':'Freedom to make life choices','Happiness.Rank':'Happiness Rank',
                         'Happiness.Score':'Happiness Score','Economy..GDP.per.Capita.':'Economy (GDP per Capita)',
                         'Health..Life.Expectancy.':'Health (Life Expectancy)'},inplace = True)

# just needed to rename dataframe for this part
df2017new = df2017


# rename columns
df2018.rename(columns = {'Overall rank':'Happiness Rank','Country or region':'Country or Region','Score':'Happiness Score',
                         'GDP per capita':'Economy (GDP per Capita)', 'Social support': 'Social Support',
                         'Healthy life expectancy':'Health (Life Expectancy)'},inplace = True)

# switch "Country or region" and "Happiness Rank"
df2018new = df2018[['Country or Region', 'Happiness Rank', 'Happiness Score','Economy (GDP per Capita)', 
                    'Social Support','Health (Life Expectancy)', 'Freedom to make life choices','Generosity','Perceptions of corruption']]

# rename columns
df2019.rename(columns = {'Overall rank':'Happiness Rank','Country or region':'Country or Region','Score':'Happiness Score',
                         'GDP per capita':'Economy (GDP per Capita)','Social support': 'Social Support',
                         'Healthy life expectancy':'Health (Life Expectancy)'},inplace = True)

# switch "Country or region" and "Happiness Rank"
df2019new = df2019[['Country or Region', 'Happiness Rank', 'Happiness Score','Economy (GDP per Capita)', 
                    'Social Support','Health (Life Expectancy)', 'Freedom to make life choices','Generosity','Perceptions of corruption']]
```

    columns are dropped!
    columns are dropped!
    columns are dropped!
    

**Assigning year to the data (only to remove it later :)):**


```python
df2015new.loc[:,"Year"] = 2015
df2016new.loc[:,"Year"] = 2016
df2017new.loc[:,"Year"] = 2017
df2018new.loc[:,"Year"] = 2018
df2019new.loc[:,"Year"] = 2019
```

    c:\python37\lib\site-packages\pandas\core\indexing.py:845: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    c:\python37\lib\site-packages\pandas\core\indexing.py:966: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    


```python
# yearsData was defined at the beginning of script
dfsNew = [df2015new.shape,df2016new.shape,df2017new.shape,df2018new.shape,df2019new.shape]

for i in range(len(dfsNew)):
  print("Current dimensions for {0} are {1}".format(yearsData[i],dfsNew[i])) 

print("")
```

    Current dimensions for 2015 are (158, 10)
    Current dimensions for 2016 are (157, 10)
    Current dimensions for 2017 are (155, 10)
    Current dimensions for 2018 are (156, 10)
    Current dimensions for 2019 are (156, 10)
    
    

**Merging Datasets:**


```python
df = pd.concat([df2015new,df2016new,df2017new,df2018new,df2019new])
df.index = np.arange(1, len(df)+1)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country or Region</th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Economy (GDP per Capita)</th>
      <th>Social Support</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Switzerland</td>
      <td>1</td>
      <td>7.587</td>
      <td>1.39651</td>
      <td>1.34951</td>
      <td>0.94143</td>
      <td>0.66557</td>
      <td>0.29678</td>
      <td>0.41978</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>2</td>
      <td>7.561</td>
      <td>1.30232</td>
      <td>1.40223</td>
      <td>0.94784</td>
      <td>0.62877</td>
      <td>0.43630</td>
      <td>0.14145</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Denmark</td>
      <td>3</td>
      <td>7.527</td>
      <td>1.32548</td>
      <td>1.36058</td>
      <td>0.87464</td>
      <td>0.64938</td>
      <td>0.34139</td>
      <td>0.48357</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Norway</td>
      <td>4</td>
      <td>7.522</td>
      <td>1.45900</td>
      <td>1.33095</td>
      <td>0.88521</td>
      <td>0.66973</td>
      <td>0.34699</td>
      <td>0.36503</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Canada</td>
      <td>5</td>
      <td>7.427</td>
      <td>1.32629</td>
      <td>1.32261</td>
      <td>0.90563</td>
      <td>0.63297</td>
      <td>0.45811</td>
      <td>0.32957</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>778</th>
      <td>Rwanda</td>
      <td>152</td>
      <td>3.334</td>
      <td>0.35900</td>
      <td>0.71100</td>
      <td>0.61400</td>
      <td>0.55500</td>
      <td>0.21700</td>
      <td>0.41100</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>779</th>
      <td>Tanzania</td>
      <td>153</td>
      <td>3.231</td>
      <td>0.47600</td>
      <td>0.88500</td>
      <td>0.49900</td>
      <td>0.41700</td>
      <td>0.27600</td>
      <td>0.14700</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>780</th>
      <td>Afghanistan</td>
      <td>154</td>
      <td>3.203</td>
      <td>0.35000</td>
      <td>0.51700</td>
      <td>0.36100</td>
      <td>0.00000</td>
      <td>0.15800</td>
      <td>0.02500</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>781</th>
      <td>Central African Republic</td>
      <td>155</td>
      <td>3.083</td>
      <td>0.02600</td>
      <td>0.00000</td>
      <td>0.10500</td>
      <td>0.22500</td>
      <td>0.23500</td>
      <td>0.03500</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>782</th>
      <td>South Sudan</td>
      <td>156</td>
      <td>2.853</td>
      <td>0.30600</td>
      <td>0.57500</td>
      <td>0.29500</td>
      <td>0.01000</td>
      <td>0.20200</td>
      <td>0.09100</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
<p>782 rows × 10 columns</p>
</div>



**Understanding High level structure of the datase:**


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 782 entries, 1 to 782
    Data columns (total 10 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   Country or Region             782 non-null    object 
     1   Happiness Rank                782 non-null    int64  
     2   Happiness Score               782 non-null    float64
     3   Economy (GDP per Capita)      782 non-null    float64
     4   Social Support                782 non-null    float64
     5   Health (Life Expectancy)      782 non-null    float64
     6   Freedom to make life choices  782 non-null    float64
     7   Generosity                    782 non-null    float64
     8   Perceptions of corruption     781 non-null    float64
     9   Year                          782 non-null    int64  
    dtypes: float64(7), int64(2), object(1)
    memory usage: 67.2+ KB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Economy (GDP per Capita)</th>
      <th>Social Support</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>781.000000</td>
      <td>782.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>78.698210</td>
      <td>5.379018</td>
      <td>0.916047</td>
      <td>1.078392</td>
      <td>0.612416</td>
      <td>0.411091</td>
      <td>0.218576</td>
      <td>0.125436</td>
      <td>2016.993606</td>
    </tr>
    <tr>
      <th>std</th>
      <td>45.182384</td>
      <td>1.127456</td>
      <td>0.407340</td>
      <td>0.329548</td>
      <td>0.248309</td>
      <td>0.152880</td>
      <td>0.122321</td>
      <td>0.105816</td>
      <td>1.417364</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.693000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2015.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>40.000000</td>
      <td>4.509750</td>
      <td>0.606500</td>
      <td>0.869363</td>
      <td>0.440183</td>
      <td>0.309768</td>
      <td>0.130000</td>
      <td>0.054000</td>
      <td>2016.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>79.000000</td>
      <td>5.322000</td>
      <td>0.982205</td>
      <td>1.124735</td>
      <td>0.647310</td>
      <td>0.431000</td>
      <td>0.201982</td>
      <td>0.091000</td>
      <td>2017.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>118.000000</td>
      <td>6.189500</td>
      <td>1.236187</td>
      <td>1.327250</td>
      <td>0.808000</td>
      <td>0.531000</td>
      <td>0.278832</td>
      <td>0.156030</td>
      <td>2018.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>158.000000</td>
      <td>7.769000</td>
      <td>2.096000</td>
      <td>1.644000</td>
      <td>1.141000</td>
      <td>0.724000</td>
      <td>0.838075</td>
      <td>0.551910</td>
      <td>2019.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Missing data:**


```python
df.isnull().sum()
```




    Country or Region               0
    Happiness Rank                  0
    Happiness Score                 0
    Economy (GDP per Capita)        0
    Social Support                  0
    Health (Life Expectancy)        0
    Freedom to make life choices    0
    Generosity                      0
    Perceptions of corruption       1
    Year                            0
    dtype: int64



We're dropping that row with missing data of Perception of Corruption.


```python
df = df.dropna()
```

## Descriptive Analytics <a id='Descriptive-Analytics'></a>

Quickly geting an idea of the overall correlation between the target variable and each input variable. Correlations are sorted in descending order. Thus, those variables at the bottom do not necessarily have the least predictive power; predictive power depends on the absolute value of correlation - generally, the larger the absolute value of correlation, the higher its predictive power.


```python
df.corr()['Happiness Score'].sort_values(ascending=False)
```




    Happiness Score                 1.000000
    Economy (GDP per Capita)        0.789719
    Health (Life Expectancy)        0.742843
    Social Support                  0.651246
    Freedom to make life choices    0.553365
    Perceptions of corruption       0.398418
    Generosity                      0.138142
    Year                            0.005946
    Happiness Rank                 -0.992053
    Name: Happiness Score, dtype: float64




```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Economy (GDP per Capita)</th>
      <th>Social Support</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Happiness Rank</th>
      <td>1.000000</td>
      <td>-0.992053</td>
      <td>-0.795110</td>
      <td>-0.647420</td>
      <td>-0.744099</td>
      <td>-0.540151</td>
      <td>-0.118290</td>
      <td>-0.372781</td>
      <td>-0.006594</td>
    </tr>
    <tr>
      <th>Happiness Score</th>
      <td>-0.992053</td>
      <td>1.000000</td>
      <td>0.789719</td>
      <td>0.651246</td>
      <td>0.742843</td>
      <td>0.553365</td>
      <td>0.138142</td>
      <td>0.398418</td>
      <td>0.005946</td>
    </tr>
    <tr>
      <th>Economy (GDP per Capita)</th>
      <td>-0.795110</td>
      <td>0.789719</td>
      <td>1.000000</td>
      <td>0.592889</td>
      <td>0.787752</td>
      <td>0.345615</td>
      <td>-0.013646</td>
      <td>0.306307</td>
      <td>0.017230</td>
    </tr>
    <tr>
      <th>Social Support</th>
      <td>-0.647420</td>
      <td>0.651246</td>
      <td>0.592889</td>
      <td>1.000000</td>
      <td>0.573252</td>
      <td>0.419795</td>
      <td>-0.037597</td>
      <td>0.126401</td>
      <td>0.368585</td>
    </tr>
    <tr>
      <th>Health (Life Expectancy)</th>
      <td>-0.744099</td>
      <td>0.742843</td>
      <td>0.787752</td>
      <td>0.573252</td>
      <td>1.000000</td>
      <td>0.341155</td>
      <td>0.010718</td>
      <td>0.250512</td>
      <td>0.130138</td>
    </tr>
    <tr>
      <th>Freedom to make life choices</th>
      <td>-0.540151</td>
      <td>0.553365</td>
      <td>0.345615</td>
      <td>0.419795</td>
      <td>0.341155</td>
      <td>1.000000</td>
      <td>0.290564</td>
      <td>0.459593</td>
      <td>0.011118</td>
    </tr>
    <tr>
      <th>Generosity</th>
      <td>-0.118290</td>
      <td>0.138142</td>
      <td>-0.013646</td>
      <td>-0.037597</td>
      <td>0.010718</td>
      <td>0.290564</td>
      <td>1.000000</td>
      <td>0.318920</td>
      <td>-0.192416</td>
    </tr>
    <tr>
      <th>Perceptions of corruption</th>
      <td>-0.372781</td>
      <td>0.398418</td>
      <td>0.306307</td>
      <td>0.126401</td>
      <td>0.250512</td>
      <td>0.459593</td>
      <td>0.318920</td>
      <td>1.000000</td>
      <td>-0.122264</td>
    </tr>
    <tr>
      <th>Year</th>
      <td>-0.006594</td>
      <td>0.005946</td>
      <td>0.017230</td>
      <td>0.368585</td>
      <td>0.130138</td>
      <td>0.011118</td>
      <td>-0.192416</td>
      <td>-0.122264</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



- Happiness Score, the target variable, has a correlation of 1 with itself (the highest possible value), which is correct.
- All other variables except Generosity appear to predict the Happiness Score with medium corealtion.
- The usefulness and independency of the the variable Generosity in predicting the target variable will be checked in feature selection techniques.

And most importantly:
-  **There are many collinear features in this dataset**



```python
plot_1= plt.subplots(figsize=(9.6,11.2))
sns.heatmap(df.corr(),fmt='d', cmap="copper", linewidths=0.6, square=True,robust=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24225860be0>




![png](/assets/img/happiness/output_28_1.png)


And to sum it up let's do a correlogram (dropping the year and happiness rank):


```python
sb.pairplot(df.drop(['Year', 'Happiness Rank'], axis = 1))
```




    <seaborn.axisgrid.PairGrid at 0x2423325a520>




![png](/assets/img/happiness/output_30_1.png)


**Inferences:**

- Most of the distributions of the variables seems normal.
- Perceptions of Corruption and Generosity variables have short range of values because the distribtution is slendered and both are slightly right tailed.
- Other variables except Perceptions of corruption and Generosity are approximately symmetrical.


## Regression Analysis <a id='regression-analysis'></a>

**Setting up data for analysis:**


```python
X = df.drop(['Country or Region','Happiness Score','Happiness Rank','Year'], axis = 1)
y = df['Happiness Score']
```

**Normalizing data:**


```python
mean = X.mean()
std = X.std()
X = (X - mean)/std
```

**Training and test data being 30% of total:**


```python
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

### Ridge Regression <a id='Ridge-Regression'></a>


**Let's define a function to give us root mean square error rmse, with a number of validation groups  = 5:**


```python
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
```


```python
ridge = Ridge()
```


```python
alphas = np.geomspace(0.01, 100, 31) #  I played to come up with this exact range for a nice graph below
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

```


```python
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation Ridge")
plt.xlabel("alpha")
plt.ylabel("rmse")
```




    Text(0, 0.5, 'rmse')




![png](/assets/img/happiness/output_42_1.png)


**Best alpha is ~22, but mattering very very little!**

Let's plot the coefficients with respect to alpha


```python
alphas = np.geomspace(0.01, 10000, 31)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.legend(X_train.columns)
```




    <matplotlib.legend.Legend at 0x2422b528850>




![png](/assets/img/happiness/output_45_1.png)



```python
np.min(cv_ridge)
```




    0.5649329314599901



Ridge Regression gets an rmse of about 0.565

Let's fit the model with alpha = 22


```python
ridge = Ridge(alpha = 22)
ridge.fit(X_train, y_train)
```




    Ridge(alpha=22)



Let's plot coefficients of each variable and predicted versus actual happiness score in the test set:


```python
ridge_coefficients = pd.DataFrame(ridge.coef_, index=X.columns, columns= ['Coefficient'])
pred_ridge = ridge.predict(X)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.25]})
ax[1].scatter(y, pred_ridge)
ax[1].set_xlabel('Actual Happiness Score')
ax[1].set_ylabel('Predicted Happiness Score')
ax[0].barh(ridge_coefficients.index, ridge_coefficients['Coefficient'])
ax[0].set_xlabel('Coefficient')
fig.set_size_inches([10, 5])
```


![png](/assets/img/happiness/output_50_0.png)


As seen, Economy has the highest effect on happiness followed by almost equally weighed Life Expectancy, Social Support and Freedom. Generosity and Perception of corruption have the least effect.

Let us now plot residual error versus happiness score:


```python
pred_ridge = ridge.predict(X)
error_ridge = pred_ridge - y
plt.scatter(y, error_ridge)
plt.xlabel('Happiness Score')
plt.ylabel('Prediction Error Ridge')
```




    Text(0, 0.5, 'Prediction Error Ridge')




![png](/assets/img/happiness/output_52_1.png)


This does not look very good, there appears to be a trend in the residual error.

### Lasso Regression <a id='Lasso-Regression'></a>

Trying Lassso similarly as above:


```python
alphas = np.geomspace(0.001,1, 101) #  I played to come up with this exact range for a nice graph below
cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]
```


```python
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Validation Lasso")
plt.xlabel("alpha")
plt.ylabel("rmse")
```




    Text(0, 0.5, 'rmse')




![png](/assets/img/happiness/output_56_1.png)


Lasso does not really provide anything useful here. The minimum rmse is obtained with an alpha = 0. This maybe is to be expected since there are few fatures.

Let's plot the coefficients with respect to alpha for academic purposes only as this will have no practical consequence.


```python
lasso = Lasso()
alphas = np.geomspace(0.001,10, 101)
coefs = []

for a in alphas:
    lasso.set_params(alpha = a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.legend(X_train.columns)
```




    <matplotlib.legend.Legend at 0x2422cb3acd0>




![png](/assets/img/happiness/output_58_1.png)


If we wanted to reduce the number of features by 2 we could set up alpha ~0.2; this however would increase rmse.


```python
np.min(cv_lasso)
```




    0.5655072419242062



Lasso gives us similar performance as Ridge (only slightly worse)


```python
lasso = Ridge(alpha = 0)
lasso.fit(X_train, y_train)
```




    Ridge(alpha=0)




```python
lasso_coefficients = pd.DataFrame(lasso.coef_, index=X.columns, columns= ['Coefficient'])
pred_lasso = lasso.predict(X)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.25]})
ax[1].scatter(y, pred_lasso)
ax[1].set_xlabel('Actual Happiness Score')
ax[1].set_ylabel('Predicted Happiness Score')
ax[0].barh(ridge_coefficients.index, ridge_coefficients['Coefficient'])
ax[0].set_xlabel('Coefficient')
fig.set_size_inches([10, 5])
```


![png](/assets/img/happiness/output_63_0.png)


Very similar to Ridge...

Let us now plot residual error versus happiness score:



```python
pred_lasso = lasso.predict(X)
error_lasso = pred_lasso - y
plt.scatter(y, error_lasso)
plt.xlabel('Happiness Score')
plt.ylabel('Prediction Error Lasso')
```




    Text(0, 0.5, 'Prediction Error Lasso')




![png](/assets/img/happiness/output_65_1.png)


Again, there appears to be a trend, however slight.

### Elastic Net Regression <a id='Elastic-Net-Regression'></a>

Let's bring in the big guns and try Elastic Net regression


```python
elastic=ElasticNet(normalize=True)
search=GridSearchCV(estimator=elastic,
                    param_grid={'alpha':np.logspace(-5,4,50),'l1_ratio':[.2,.4,.6,.8]},
                    scoring='neg_mean_squared_error',n_jobs=1,refit=True,cv=10)
```

Let's find the best fitting:


```python
search.fit(X_train,y_train)
search.best_params_
```




    {'alpha': 8.286427728546843e-05, 'l1_ratio': 0.2}




```python
abs(search.best_score_)**0.5
```




    0.562796603750156



A very, very tiny improvement compared to Lasso and Ridge but here it is!


```python
elastic=ElasticNet(normalize=True,alpha=8.286427728546843e-05,l1_ratio=0.2)
elastic.fit(X_train,y_train)

```




    ElasticNet(alpha=8.286427728546843e-05, l1_ratio=0.2, normalize=True)




```python
elastic_coefficients = pd.DataFrame(elastic.coef_, index=X.columns, columns= ['Coefficient'])
pred_elastic = elastic.predict(X)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.25]})
ax[1].scatter(y, pred_elastic)
ax[1].set_xlabel('Actual Happiness Score')
ax[1].set_ylabel('Predicted Happiness Score')
ax[0].barh(ridge_coefficients.index, ridge_coefficients['Coefficient'])
ax[0].set_xlabel('Coefficient')
fig.set_size_inches([10, 5])
```


![png](/assets/img/happiness/output_74_0.png)


Almost identical to previous!


```python
error_elastic = pred_elastic - y
plt.scatter(y, error_elastic)
plt.xlabel('Happiness Score')
plt.ylabel('Prediction Error Elastic')
```




    Text(0, 0.5, 'Prediction Error Elastic')




![png](/assets/img/happiness/output_76_1.png)


### Summary <a id='Summary'></a>

Turns out we can predict mean happiness score of people in a location with an error of about 0.55 as per the above scoring criteria. Ridge and Lasso gave very similar results with Elastic Net Regression model giving only a slight improvement.

Happiness is affected the most by Economic Wellbeing, followed by Health, Social Support and Freedom. 

Even this simple dataset with not too many features it is not trivial to deal with collinearity or with non-linear effects. 2nd, 3rd and even 4th order polynomial fits were attempted with no improvements. It may be that another method may work better. I am planning to apply PCA, neural networks and tree models and compare their performance againts the regression benchmark.

Till then, prosper, stay healthy and be happy!


```python

```


{% include button.html text="Github" icon="github" link="https://github.com/flikrama" color="#0366d6" %} {% include button.html text="Linkedin" icon="linkedin" link="https://www.linkedin.com/in/likrama/" color="#0e76a8" %}   [**Resume**](/assets/resume/Fatmir_Likrama.pdf)