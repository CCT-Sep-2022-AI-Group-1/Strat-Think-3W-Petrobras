# A realistic and public dataset with rare undesirable real events in oil wells

## Abstract
* Detection of **undesirable events** in oil and gas wells can help prevent production losses, environmental accidents, and human casualties and reduce maintenance costs.
* Dataset
	* instances of 8 types of undesirable events characterized by 8 process variables.
	* It can be used as a benchmark for the development of ML techniques


## Introduction

### Undesirable events: 
* Examples
	* flow influx detection during drilling; 
	* leak detection and location in water and oil pipelines; 
	* fault detection in industrial plants, in oil wells, in Electrical Submersible Pumps, in gas compressor valves
	* abnormal event detection in oil wells, EEG, ECG
	* power-quality disturbance detection
	* handgun detection

* Abnormal Event Management (AEM): responding to abnormal events
	* Diagnosis in automated AEM can be viewed as a **classification problem**
	
* [2010 BP oil spill](https://en.wikipedia.org/wiki/Deepwater_Horizon_oil_spill) as example of potential magnitude of losses and costs

### AEM Project (MAE)
* Mid-2017: **Petrobras AEM project (MAE)** for oil and gas wells
	* Operational Unit in Brazilian state of Espírito Santo (UO-ES)
	* complement the current system based on alarms with ML applied to a vast Multivariate Time Series (MTS)
	* develop a new automated AEM to detect and classify occurrences of **8 types of undesirable events** in offshore naturally flowing wells that are in a normal state in a shorter time with better performance
	* ML algorithms have been shows to be a suitable search strategy
* These 8 selected types of undesirable events had been responsible for most of the production loss in the UO- ES in the last years
* No public or private dataset with enough undesirable events in terms of quantity and diversity in oil and gas wells has been found so far.
* As a result, it was decided to generate a dataset to be used in the development of automated AEM with ML. 
	* 3W dataset: can be used for development of ML techniques related to inherent difficulties of actual data  
	* Some possibilities:
		* preprocessing, filters, transformations
		* classifiers based on **trees**, **artificial neural networks**, distances, ensembles, etc

## Background

### Offshore Naturally Flowing Wells
* Oil Well: set of sensors and systems installed on the seadbed, downhole (the well itself), or on the surface
* Naturally Flowing wells tends to have less equipment, instrumentation, control loops and automation

#### Offshore scenario (see Figure 1):
* Production tubing
* Production Line
* Electro-Hydraulic Umbilical
* Platform (ship)
* subsea Christmas Tree (seabed)
* Permanent Downhole Gauge (PDG): pressure sensors, fixed at the Production tubing
* Temperature and Pressure Transducer (TPT): temperature sensors, part of Christmas Tree
* Downhole Safety Valve (DHSV a.k.a. DSV)
* Production Choke (PCK)

#### Normal State flowing wells
* MAE focus on offshore naturally flowing wells in _normal state_, 
	* undesirable events present in 3W dataset started from a _normal state_
	* Other states: _closed-in_ (no flow at all) and _starting-up_ (between closed-in and normal state)
* Cost-Benefit Ratio of having instrumentation in certain positions and reliability of instrumentation despite of environment hostility
	* Pressure at the PDG	 
	* Pressure at the TPT
	* Temperature at the TPT
	* Pressure upstream of the PCK
	* Temperature downstream of the PCK

### Types of undesirable events in oil wells
**Important:** there is not always consensus regarding names and what
these types of undesirable events mean, even among experts

#### 1) Abrupt Increase of Basic Sediment and Water (BSW)
* BSW: Ratio between Water and Sediment flow rate X the liquid flow rate, under normal temperature and pressure (NTP)
* Expected to increase due to increased water production from either natural reservoir aquifer or artificial injection to avoid declining production
* Sudden increase of BSW can lead to several problems

#### 2) Spurious Closure of DHSV
* DHSV is installed in the production tubing of wells
* Must ensure closing of the well if:
	* production unit and well are physically disconnected
	* emergency of catastrophic failure of surface equipment
* These closure functions fails unexpectedly, often w/out any indication on the surface (e.g. pressure drop in the hydraulic actuator)

#### 3) Severe Slugging
* Critical type of instability
* [Video here (watch 2x)](https://www.youtube.com/watch?v=S3jNx8ZlAxQ)
* [Longer video here](https://www.youtube.com/watch?v=GyRoigEyITk)
* Well-defined periodicity (30/45/60 min)
* Intense enough to be detected along the entire production line
* Stress or even damage equipment in the well and/or industrial plant

#### 4) Flow Instability
* Similar to Severe Slugging, but without periodicity and with tolerable amplitudes
* it can progress to severe slugging

#### 5) Rapid Productivity Loss
* depends on several properties: static pressure reservoir, percentage of BSW, viscosity of the produced fluid, diameter of the production line, etc
* Changes on these properties may make system's energy no longer sufficient to overcome losses (flow slows/stops)

#### 6) Quick Restriction in PCK
* PCK is installed at the beginning of the production unit
* responsible for well control at the surface
* According Petrobras, it occurs with an amplitude above a certain reference and in short time, e.g. 5% in less than 10s

#### 7) Scaling in PCK
* Suspectibility of inorganic deposits which can dramatically reduce oil and gas production

#### 8) Hydrate in Production Line
* Crystalline compound from water + natural gas + high pressures + low temperatures
* More frequent in gas pipelines and gas producing wells
* It can also be formed in oil-producing wells, to the point of totally interrupting their flow


### Response times
To confirm real occurences of each event, professionals analyze time windows of different sizes:
<table>
	<tr>
		<td>**Undesirable Event**</td>
    	<td>**Window Size**</td>
   	</tr>
	<tr>
		<td>1) Abrupt Increase of BSW</td>
    	<td>12 hours</td>
	</tr>
	<tr>
		<td>2) Spurious Closure of DHSV</td>
    	<td>5 min-20 min</td>
	</tr>
	<tr>
		<td>3) Severe Slugging</td>
    	<td>5 hours</td>
	</tr>
	<tr>
		<td>4) Flow Instability</td>
    	<td>15 min</td>
	</tr>
	<tr>
		<td>5) Rapid Productivity Loss</td>
    	<td>12 hours</td>
	</tr>
	<tr>
		<td>6) Quick Restriction in PCK</td>
    	<td>15 mins</td>
	</tr>
	<tr>
		<td>7) Scaling in PCK</td>
    	<td>72 hours</td>
	</tr>
	<tr>
		<td>8) Hydrate in Production Line</td>
    	<td>30 min-5 hours</td>
	</tr>	
</table>
	
### Machine Learning
* Predictive algorithms that perform classification
* Input: MTS variables acquired from processes
* Output: Types of undesirable events
* The amount of algorithm possibilities applied here goes beyond the scope of this article


### Multivariate Time Series (MTS)
* Dataset DS is a set of m MTS
* Each MTS i in an instance composed of a set of n univariate time series


## 3W Dataset preparation
### Considerations
* Real instance vs simulated and hand-drawn instances: decreased imbalance for knowledge
* Two types of labeling on each undesirable event
	* 1) Each instance has single code associated with normal operation or some code associated w/ undesirable event at that instant
		* no instance contains more than one undesirable event. 
		* it provides a grouping of instances depending on the type of undesirable event they contain
		* it allows the development of offline classifiers, those that do not aim to estimate when the event started or ended inside each instance. 
	* 2) Observation's level, a single code associated with normal operation or some code associated w/ undesirable event at that instant
		* Any type has up to 3 periods: 
			* normal: no evidence of any type of anomaly
			* faulty transient: undesirable event are still ongoing
			* faulty steady state
		* This strategy provides the possibility of early classification.
			* faulty transient periods can be learned
			* predicts the period faulty steady state
* No preprocessing for "realistic aspects": NaN values, frozen variables, instances with different sizes and outliers
	* hand-drawn is free of such problems
* Simulated instances obtained with flow simulator (OLGA, a standard tool)
* Hand-drawn instances created from Petrobras experts

### Files
* All CSVs were grouped into directories based on the instance label
* All instances were generated with observations obtained with a fixed sampling rate (1 Hz). 
* Only the following units were used: 
	* Pascal [Pa], 
	* standard cubic meters per second [sm3/s], 
	* degrees Celsius [oC]. 
* The source of each instance was incorporated on the name of its CSV file. 
* All actual well names were replaced by generic names as a requirement of Petrobras for the 3W dataset's publication.
* Instances:
<table>
	<tr>
		<td>**Type of Event**</td>
    	<td>**Real**</td>
    	<td>**Simulated**</td>
    	<td>**Hand-Drawn**</td>
    	<td>**TOTAL**</td>
   	</tr>
	<tr>
		<td>0 - Normal</td>
    	<td>597</td>
    	<td>-</td>
    	<td>-</td>
    	<td>597</td>
	</tr>
	<tr>
		<td>1 - Abrupt Increase BSW</td>
    	<td>5</td>
    	<td>114</td>
    	<td>19</td>
    	<td>129</td>
	</tr>
	<tr>
		<td>2 - Spurious Closure of DHSV</td>
    	<td>22</td>
    	<td>16</td>
    	<td>-</td>
    	<td>38</td>
	</tr>
	<tr>
		<td>3 - Severe Slugging</td>
    	<td>32</td>
    	<td>74</td>
    	<td>-</td>
    	<td>106</td>
	</tr>
	<tr>
		<td>4 - Flow Instability</td>
    	<td>344</td>
    	<td>-</td>
    	<td>-</td>
    	<td>344</td>
	</tr>
	<tr>
		<td>5 - Rapid Productivity Loss</td>
    	<td>12</td>
    	<td>439</td>
    	<td>-</td>
    	<td>451</td>
	</tr>
	<tr>
		<td>6 - Quick Restriction in PCK</td>
    	<td>6</td>
    	<td>215</td>
    	<td>-</td>
    	<td>221</td>
	</tr>
	<tr>
		<td>7 - Scaling in PCK</td>
    	<td>4</td>
    	<td>-</td>
    	<td>10</td>
    	<td>14</td>
	</tr>
	<tr>
		<td>8 - Hydrate in Production Line</td>
    	<td>3</td>
    	<td>81</td>
    	<td>-</td>
    	<td>84</td>
	</tr>
	<tr>
	   <td>**TOTAL**</td>
		<td>**1025**</td>
    	<td>**939**</td>
    	<td>**20**</td>
    	<td>**1984**</td>
	</tr>
</table>

* columns [defined here](https://github.com/petrobras/3W/blob/master/overviews/_baseline/main.ipynb):
	* timestamp: observations timestamps loaded into pandas DataFrame as its index;
	* P-PDG: pressure variable at the Permanent Downhole Gauge (PDG);
	* P-TPT: pressure variable at the Temperature and Pressure Transducer (TPT);
	* T-TPT: temperature variable at the Temperature and Pressure Transducer (TPT);
	* P-MON-CKP: pressure variable upstream of the production choke (CKP);
	* T-JUS-CKP: temperature variable downstream of the production choke (CKP);
	* P-JUS-CKGL: pressure variable upstream of the gas lift choke (CKGL);
	* T-JUS-CKGL: temperature variable upstream of the gas lift choke (CKGL);
	* QGL: gas lift flow rate;
	* class: observations labels associated with three types of periods (normal, fault transient, and faulty steady state).

### Dificulties
* Total: 50913215 observations of 15872 variables in 1984 instances
* Missing variables: 4947 (31.17%) due to sensor/network issues
* Frozen variables: 1535 (9.67%) integer value or single float is considered frozen, not necessarily a problem, but symptom of issues
* Unlabeled observations: 5130 (0.01% of all observations)
	* Even with this percentage, some technique should be used to treat the unlabeled observations. 

















