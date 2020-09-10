---
title: The happiness project
categories:
- Linear Regression

---
### Ahh happiness
How to be happy and other stuff...|
How to be happy and other stuff...|
How to be happy and other stuff...|

How to be happy and other stuff...|
How to be happy and other stuff...|

How to be happy and other stuff...|
How to be happy and other stuff...|

How to be happy and other stuff...|

```python
import numpy as np
import pandas as pd

```


```python
df = pd.read_csv('datasets_12603_17232_Life Expectancy Data.csv')
```


```python
df.head()
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
      <th>Year</th>
      <th>Status</th>
      <th>Life expectancy</th>
      <th>Adult Mortality</th>
      <th>infant deaths</th>
      <th>Alcohol</th>
      <th>percentage expenditure</th>
      <th>Hepatitis B</th>
      <th>Measles</th>
      <th>...</th>
      <th>Polio</th>
      <th>Total expenditure</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>thinness  1-19 years</th>
      <th>thinness 5-9 years</th>
      <th>Income composition of resources</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2015</td>
      <td>Developing</td>
      <td>65.0</td>
      <td>263.0</td>
      <td>62</td>
      <td>0.01</td>
      <td>71.279624</td>
      <td>65.0</td>
      <td>1154</td>
      <td>...</td>
      <td>6.0</td>
      <td>8.16</td>
      <td>65.0</td>
      <td>0.1</td>
      <td>584.259210</td>
      <td>33736494.0</td>
      <td>17.2</td>
      <td>17.3</td>
      <td>0.479</td>
      <td>10.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2014</td>
      <td>Developing</td>
      <td>59.9</td>
      <td>271.0</td>
      <td>64</td>
      <td>0.01</td>
      <td>73.523582</td>
      <td>62.0</td>
      <td>492</td>
      <td>...</td>
      <td>58.0</td>
      <td>8.18</td>
      <td>62.0</td>
      <td>0.1</td>
      <td>612.696514</td>
      <td>327582.0</td>
      <td>17.5</td>
      <td>17.5</td>
      <td>0.476</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2013</td>
      <td>Developing</td>
      <td>59.9</td>
      <td>268.0</td>
      <td>66</td>
      <td>0.01</td>
      <td>73.219243</td>
      <td>64.0</td>
      <td>430</td>
      <td>...</td>
      <td>62.0</td>
      <td>8.13</td>
      <td>64.0</td>
      <td>0.1</td>
      <td>631.744976</td>
      <td>31731688.0</td>
      <td>17.7</td>
      <td>17.7</td>
      <td>0.470</td>
      <td>9.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>Developing</td>
      <td>59.5</td>
      <td>272.0</td>
      <td>69</td>
      <td>0.01</td>
      <td>78.184215</td>
      <td>67.0</td>
      <td>2787</td>
      <td>...</td>
      <td>67.0</td>
      <td>8.52</td>
      <td>67.0</td>
      <td>0.1</td>
      <td>669.959000</td>
      <td>3696958.0</td>
      <td>17.9</td>
      <td>18.0</td>
      <td>0.463</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>Developing</td>
      <td>59.2</td>
      <td>275.0</td>
      <td>71</td>
      <td>0.01</td>
      <td>7.097109</td>
      <td>68.0</td>
      <td>3013</td>
      <td>...</td>
      <td>68.0</td>
      <td>7.87</td>
      <td>68.0</td>
      <td>0.1</td>
      <td>63.537231</td>
      <td>2978599.0</td>
      <td>18.2</td>
      <td>18.2</td>
      <td>0.454</td>
      <td>9.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python

```


```python
df.keys()
```




    Index(['Country', 'Year', 'Status', 'Life expectancy ', 'Adult Mortality',
           'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
           'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
           'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
           ' thinness  1-19 years', ' thinness 5-9 years',
           'Income composition of resources', 'Schooling'],
          dtype='object')




```python
null_columns = df.columns[df.isnull().any()]
null0 = df[df.isnull().any(axis=1)][null_columns].head()

```


```python
le = np.array(df['Life expectancy '])
```


```python
np.histogram?
```


```python
df.isnull()
```


```python
null0
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
      <th>Life expectancy</th>
      <th>Adult Mortality</th>
      <th>Alcohol</th>
      <th>Hepatitis B</th>
      <th>BMI</th>
      <th>Polio</th>
      <th>Total expenditure</th>
      <th>Diphtheria</th>
      <th>GDP</th>
      <th>Population</th>
      <th>thinness  1-19 years</th>
      <th>thinness 5-9 years</th>
      <th>Income composition of resources</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>75.6</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>95.0</td>
      <td>59.5</td>
      <td>95.0</td>
      <td>NaN</td>
      <td>95.0</td>
      <td>4132.762920</td>
      <td>39871528.0</td>
      <td>6.0</td>
      <td>5.8</td>
      <td>0.743</td>
      <td>14.4</td>
    </tr>
    <tr>
      <th>44</th>
      <td>71.7</td>
      <td>146.0</td>
      <td>0.34</td>
      <td>NaN</td>
      <td>47.0</td>
      <td>87.0</td>
      <td>3.60</td>
      <td>87.0</td>
      <td>294.335560</td>
      <td>3243514.0</td>
      <td>6.3</td>
      <td>6.1</td>
      <td>0.663</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>45</th>
      <td>71.6</td>
      <td>145.0</td>
      <td>0.36</td>
      <td>NaN</td>
      <td>46.1</td>
      <td>86.0</td>
      <td>3.73</td>
      <td>86.0</td>
      <td>1774.336730</td>
      <td>3199546.0</td>
      <td>6.3</td>
      <td>6.2</td>
      <td>0.653</td>
      <td>11.1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>71.4</td>
      <td>145.0</td>
      <td>0.23</td>
      <td>NaN</td>
      <td>45.3</td>
      <td>89.0</td>
      <td>3.84</td>
      <td>89.0</td>
      <td>1732.857979</td>
      <td>31592153.0</td>
      <td>6.4</td>
      <td>6.3</td>
      <td>0.644</td>
      <td>10.9</td>
    </tr>
    <tr>
      <th>47</th>
      <td>71.3</td>
      <td>145.0</td>
      <td>0.25</td>
      <td>NaN</td>
      <td>44.4</td>
      <td>86.0</td>
      <td>3.49</td>
      <td>86.0</td>
      <td>1757.177970</td>
      <td>3118366.0</td>
      <td>6.5</td>
      <td>6.4</td>
      <td>0.636</td>
      <td>10.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df)
```

              Country  Year      Status  Life expectancy   Adult Mortality  \
    0     Afghanistan  2015  Developing              65.0            263.0   
    1     Afghanistan  2014  Developing              59.9            271.0   
    2     Afghanistan  2013  Developing              59.9            268.0   
    3     Afghanistan  2012  Developing              59.5            272.0   
    4     Afghanistan  2011  Developing              59.2            275.0   
    ...           ...   ...         ...               ...              ...   
    2933     Zimbabwe  2004  Developing              44.3            723.0   
    2934     Zimbabwe  2003  Developing              44.5            715.0   
    2935     Zimbabwe  2002  Developing              44.8             73.0   
    2936     Zimbabwe  2001  Developing              45.3            686.0   
    2937     Zimbabwe  2000  Developing              46.0            665.0   
    
          infant deaths  Alcohol  percentage expenditure  Hepatitis B  Measles   \
    0                62     0.01               71.279624         65.0      1154   
    1                64     0.01               73.523582         62.0       492   
    2                66     0.01               73.219243         64.0       430   
    3                69     0.01               78.184215         67.0      2787   
    4                71     0.01                7.097109         68.0      3013   
    ...             ...      ...                     ...          ...       ...   
    2933             27     4.36                0.000000         68.0        31   
    2934             26     4.06                0.000000          7.0       998   
    2935             25     4.43                0.000000         73.0       304   
    2936             25     1.72                0.000000         76.0       529   
    2937             24     1.68                0.000000         79.0      1483   
    
          ...  Polio  Total expenditure  Diphtheria    HIV/AIDS         GDP  \
    0     ...    6.0               8.16         65.0        0.1  584.259210   
    1     ...   58.0               8.18         62.0        0.1  612.696514   
    2     ...   62.0               8.13         64.0        0.1  631.744976   
    3     ...   67.0               8.52         67.0        0.1  669.959000   
    4     ...   68.0               7.87         68.0        0.1   63.537231   
    ...   ...    ...                ...          ...        ...         ...   
    2933  ...   67.0               7.13         65.0       33.6  454.366654   
    2934  ...    7.0               6.52         68.0       36.7  453.351155   
    2935  ...   73.0               6.53         71.0       39.8   57.348340   
    2936  ...   76.0               6.16         75.0       42.1  548.587312   
    2937  ...   78.0               7.10         78.0       43.5  547.358879   
    
          Population   thinness  1-19 years   thinness 5-9 years  \
    0     33736494.0                   17.2                 17.3   
    1       327582.0                   17.5                 17.5   
    2     31731688.0                   17.7                 17.7   
    3      3696958.0                   17.9                 18.0   
    4      2978599.0                   18.2                 18.2   
    ...          ...                    ...                  ...   
    2933  12777511.0                    9.4                  9.4   
    2934  12633897.0                    9.8                  9.9   
    2935    125525.0                    1.2                  1.3   
    2936  12366165.0                    1.6                  1.7   
    2937  12222251.0                   11.0                 11.2   
    
          Income composition of resources  Schooling  
    0                               0.479       10.1  
    1                               0.476       10.0  
    2                               0.470        9.9  
    3                               0.463        9.8  
    4                               0.454        9.5  
    ...                               ...        ...  
    2933                            0.407        9.2  
    2934                            0.418        9.5  
    2935                            0.427       10.0  
    2936                            0.427        9.8  
    2937                            0.434        9.8  
    
    [2938 rows x 22 columns]
    


```python
np.histogram(le[~np.isnan(le)])
```




    (array([  4,  50, 134, 229, 277, 405, 580, 819, 362,  68], dtype=int64),
     array([36.3 , 41.57, 46.84, 52.11, 57.38, 62.65, 67.92, 73.19, 78.46,
            83.73, 89.  ]))




```python
import matplotlib.pyplot as plt
```


```python
plt.hist(le[~np.isnan(le)])
```




    (array([  4.,  50., 134., 229., 277., 405., 580., 819., 362.,  68.]),
     array([36.3 , 41.57, 46.84, 52.11, 57.38, 62.65, 67.92, 73.19, 78.46,
            83.73, 89.  ]),
     <a list of 10 Patch objects>)




![alt text](/assets/img/happiness/Project_1.0_13_1.png)



```python
df.keys()
```




    Index(['Country', 'Year', 'Status', 'Life expectancy ', 'Adult Mortality',
           'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
           'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
           'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
           ' thinness  1-19 years', ' thinness 5-9 years',
           'Income composition of resources', 'Schooling'],
          dtype='object')




```python
df.plot.scatter(x ='Year', y ='Life expectancy ', c = 'Country'  )
```


```python
df.plot.scatter?
```


```python
import seaborn
```


```python
fg = seaborn.scatterplot(x=df['percentage expenditure'],y = df['Life expectancy '], hue=df['Status'], hue_order=np.unique(df['Status']))

```


![alt text](/assets/img/happiness/Project_1.0_18_0.png)



```python
plt.hist(df['Adult Mortality'])
```

    c:\python37\lib\site-packages\numpy\lib\histograms.py:839: RuntimeWarning: invalid value encountered in greater_equal
      keep = (tmp_a >= first_edge)
    c:\python37\lib\site-packages\numpy\lib\histograms.py:840: RuntimeWarning: invalid value encountered in less_equal
      keep &= (tmp_a <= last_edge)
    




    (array([729., 765., 610., 423., 176., 117.,  54.,  24.,  17.,  13.]),
     array([  1. ,  73.2, 145.4, 217.6, 289.8, 362. , 434.2, 506.4, 578.6,
            650.8, 723. ]),
     <a list of 10 Patch objects>)




![alt text](/assets/img/happiness/Project_1.0_19_2.png)



```python
print(df['Adult Mortality'])
```

    0       263.0
    1       271.0
    2       268.0
    3       272.0
    4       275.0
            ...  
    2933    723.0
    2934    715.0
    2935     73.0
    2936    686.0
    2937    665.0
    Name: Adult Mortality, Length: 2938, dtype: float64
    


```python
df['percentage expenditure'][1:10]
```




    1    73.523582
    2    73.219243
    3    78.184215
    4     7.097109
    5    79.679367
    6    56.762217
    7    25.873925
    8    10.910156
    9    17.171518
    Name: percentage expenditure, dtype: float64

{% include button.html text="My github" icon="github" link="https://github.com/flikrama" color="#0366d6" %} {% include button.html text="My Linkedin" icon="linkedin" link="https://www.linkedin.com/in/likrama/" color="#0e76a8" %} [**Resume**](/assets/resume/Fatmir_Likrama.pdf)