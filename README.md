# Activity Aware Pulse Rate Algorithm Project

In this repository we create a **Pulse Rate Algorithm** based on a Random Forest Regressor model.
Then the Pulse Rate Algorithm is applied on a **Clinical Application** to discover healthcare trends.

## Introduction
A core feature that many users expect from their wearable devices is pulse rate estimation. Continuous pulse rate estimation can be informative for many aspects of a wearer's health. Pulse rate during exercise can be a measure of workout intensity and resting heart rate is sometimes used as an overall measure of cardiovascular fitness. We will create a pulse rate estimation algorithm for a wrist-wearable device.
  
## Physiological Mechanics of Pulse Rate Estimation
Pulse rate is typically estimated by using the PPG (photoplethysmogram) sensor. When the ventricles contract, the capilaries in the wrist fill with blood. The (typically green) light emitted by the PPG sensor is absorbed by red blood cells in these capilaries and the photodetector will see the drop in reflected light. When the blood returns to the heart, fewer red blood cells in the wrist absorb the light and the photodetector sees an increase in reflected light. The period of this oscillating waveform is the pulse rate.

![PPG Mechanics](imgs/ppg_mechanics.png)
  
However, the heart beating is not the only phenomenon that modulates the PPG signal. Blood in the wrist is fluid, and arm movement will cause the blood to move correspondingly. During exercise, like walking or running, we see another periodic signal in the PPG due to this arm motion. Our pulse rate estimator  has to be careful not to confuse this periodic signal with the pulse rate.  
  
We can use the accelerometer signal of our wearable device to help us keep track of which periodic signal is caused by motion. Because the accelerometer is only sensing arm motion, any periodic signal in the accelerometer is likely not due to the heart beating, and only due to the arm motion. If our pulse rate estimator is picking a frequency that's strong in the accelerometer, it may be making a mistake.  
  
All estimators will have some amount of error. How much error is tolerable depends on the application. If we were using these pulse rate estimates to compute long term trends over months, then we may be more robust to higher error variance. However, if we wanted to give information back to the user about a specific workout or night of sleep, we would require a much lower error. 

## Algorithm Confidence and Availability
Many machine learning algorithms produce outputs that can be used to estimate their per-result error. For example in logistic regression you can use the predicted class probabilities to quantify trust in the classification. A classification where one class has a very high probability is probably more accurate than one where all classes have similar probabilities. Certainly, this method is not perfect and won't perfectly rank-order estimates based on error. But if accurate enough, it allows consumers of the algorithm more flexibility in how to use it. We call this estimation of the algorithms error the *confidence*. 

In pulse rate estimation, having a confidence value can be useful if a user wants just a handful of high-quality pulse rate estimate per night. They can use the confidence algorithm to select the 20 most confident estimates at night and ignore the rest of the outputs. Confidence estimates can also be used to set the point on the error curve that we want to operate at by sacrificing the number of estimates that are considered valid. There is a trade-off between *availability* and error. For example if we want to operate at 10% availability, we look at our training dataset to determine the condince threshold for which 10% of the estimates pass. Then only if an estimate's confidence value is above that threshold do we consider it valid. See the error vs. availability curve below.

![Error vs. Availability](imgs/error_vs_availability.png)

This plot is created by computing the mean absolute error at all -- or at least 100 of -- the confidence thresholds in the dataset.

Building a confidence algorithm for pulse rate estimation is a little tricker than logistic regression because intuitively there isn't some transformation of the algorithm output that can make a good confidence score. However, by understanding our algorithm behavior we can come up with some general ideas that might create a good confidence algorithm. For example, if our algorithm is picking a strong frequency component that's not present in the accelerometer we can be relatively confidence in the estimate. We turn this idea into an algorithm by quantifying "strong frequency component".

-----
## Pulse Rate Algorithm Overview
  
### Algorithm Specifications
The model consists of a Random Forest Regressor model which estimates pulse rate from the PPG signal and a 3-axis accelerometer signals.
It is assumed that pulse rate will be restricted between 40BPM (beats per minute) and 240BPM.
The algorithm produces an estimation confidence. A higher confidence value means that this estimate should be more accurate than an estimate with a lower confidence value.
The algorithm is able to produces an output at least every 2 seconds.

For a more detailed description of the algorithm see file `pulse_rate_estimation.ipynb`.

### Success Criteria
The pre-defined success criteria for the algorithm performance is as follows: the mean absolute error at 90% availability must be less than 15 BPM on the test set.  Put another way, the best 90% of your estimates--according to the confidence output (see below)-- must have a mean absolute error of less than 15 BPM.

### Some important processing steps
  1. We bandpass filter all signals. We use the 40-240BPM range to create the pass band.
  2. We use spectrograms (plt.specgram) to visualize the signals in the frequency domain. We plot the estimates on top of the spectrogram to see where things are going wrong.
  3. The confidence of the model estimates are computed by summing frequency spectrum near the pulse rate estimate and dividing it by the sum of the entire spectrum. The idea is to consider the frequency spectrum and ask "How much energy in the frequency spectrum is concentrated near the pulse rate estimate?"
  
### Dataset
We use the Troika<sup>1</sup> dataset to build the algorithm. Find the dataset under datasets/troika/training_data. The README in that folder contains some more information on how to interpret the data.

1. **Troika** - Zhilin Zhang, Zhouyue Pi, Benyuan Liu, ‘‘TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type Photoplethysmographic Signals During Intensive Physical Exercise,’’IEEE Trans. on Biomedical Engineering, vol. 62, no. 2, pp. 522-531, February 2015. Link

-----
## Clinical Application Simulation

After a pulse rate algorithm has been developed, we can use it to compute more clinically meaningful features and discover healthcare trends.

Specifically, we use 24 hours of heart rate data from 1500 samples to try to validate the well known trend that average resting heart rate increases up until middle age and then decreases into old age. We'll also see if resting heart rates are higher for women than men. This trend is illustrated in this image:

![heart-rate-age-ref-chart](imgs/heart-rate-age-reference-chart.jpg)

### Dataset (CAST)

The data comes from the [Cardiac Arrythmia Suppression Trial (CAST)](https://physionet.org/content/crisdb/1.0.0/), which was sponsored by the National Heart, Lung, and Blood Institute (NHLBI). CAST collected 24 hours of heart rate data from ECGs from people who have had a myocardial infarction (MI) within the past two years.<sup>2</sup> This data has been smoothed and resampled to more closely resemble PPG-derived pulse rate data from a wrist wearable.<sup>3</sup>

-----
## Repository file descriptions

These are the files in this repo:
- `README.md`
- `datasets` - direcory containing the datasets. 
- `pulse_rate_estimation.ipynb`<sup>*</sup> - pulse rate algorithm and write-up
- `etl.py` - ETL helper functions
- `ml.py` - machine learning model related helper functions
- `plot_util.py` - plotting helper functions
- `clinical_application_simulation.ipynb` - simple clinical application example
-----
## Citations
1. **Troika** - Zhilin Zhang, Zhouyue Pi, Benyuan Liu, ‘‘TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type Photoplethysmographic Signals During Intensive Physical Exercise,’’IEEE Trans. on Biomedical Engineering, vol. 62, no. 2, pp. 522-531, February 2015. Link
2. **CAST RR Interval Sub-Study Database Citation** - Stein PK, Domitrovich PP, Kleiger RE, Schechtman KB, Rottman JN. Clinical and demographic determinants of heart rate variability in patients post myocardial infarction: insights from the Cardiac Arrhythmia Suppression Trial (CAST). Clin Cardiol 23(3):187-94; 2000 (Mar)
3. **Physionet Citation** - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals (2003). Circulation. 101(23):e215-e220.


*This project is part of the Udacity Nanodegree programm "AI in Healthcare" (November 2020).*

