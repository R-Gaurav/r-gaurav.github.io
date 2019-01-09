---
tags: research paper
---

This short blog introduces the Open-source project "Train Delay Estimation" to estimate train delays in India. It is first of its kind and the [research paper](https://ieeexplore.ieee.org/document/8570014) has been accepted at IEEE ITSC-2018.

In India, most of us have travelled via trains at one point or another in life. Due to large and complex railway networks, trains often get delayed at stations. What if there was a way we could predict the delays of trains at their in-line stations? This would certainly help us in planning our journey well and also the businesses which depend on railways.

Therefore we designed an N-Order Markov Late Minutes Prediction Framework (N-OMLMPF) using Random Forest Regressors (and Ridge Regressors) to predict the late minutes at a desired in-line station, given a train number and its journey route information. The Regressors learned from past journey data of trains. In fact, the N-OMLMPF was developed to predict the delays of those trains too which were not used for training the Regressors, thus the prediction framework was made train-generic.

Note: Details presented here are very brief and more information could be found at the [arxiv version of our paper](https://arxiv.org/pdf/1806.02825.pdf).

## Data and Analysis
We collected journey data for 135 trains from [railwayapi.com](https://railwayapi.com) over a period of 2 years; from March 2016 to February 2018. After pre-processing, we found that delay at a station is governed by delays at N number of stations previous to it, the month of journey, the type of train etc..

## Regressor models training
Since late minutes at a station are continuous values, we opted for Regressor models to learn from past data of delays at stations. _That's fine, but what's Markov doing here?_ Well... as mentioned earlier, delay at a station depends on N number of stations previous to it, and this is where _Markov_ comes in picture. As per Markov Process definition, the outcome is dependent only on current state and not on any previous states. However if the outcome is dependent on N previous states, it's called N-Order Markov Process.

## N-OMLMPF
An N-Order Markov Late Minutes Prediction Framework was devised where we used the trained Regressor models to predict late minutes at a station. The late minutes predicted for N stations previous to a current station is fed forward into the framework. We employed station features (geographical location, degree of connectivity and strength of traffic) to make N-OMLMPF train-generic.

## Results
We evaluated the late minutes prediction by using the Confidence Interval metrics (68%, 95% and 99%) and Root Mean Square Error. Under 95% Confidence Interval we achieved 62% accuracy, details of which could be found in our paper.

## Interested to contribute?
We would highly appreciate your contributions to scale this framework India-wide. Please visit [our github page](https://github.com/R-Gaurav/train-delay-estimation) for more information.

Thank you for stopping by!
