# Overview
## Task Description
The task data we are analyzing comes from a digitized version of the traditional paper and pencil A/B Trails cognitive task. The users are instructed to connect a series of labeled circles on the screen in a particular order using a stylus on an iPad. The user progresses through four 'trails': trails 1 and 3 are short 'practice' screens for the actual sequential and alternating test conditions on trails 2 and 4 - typically called TrailsA and TrailsB, respectively. We will not analyze the practice trails 1 or 3. 
## Data Description
### Behavioral Data:
The behavioral data are stored as csv files in preprocessed/. The csvs contain x and y points sampled at ~60Hz, along with the correct_path, the actual_path drawn, the timestamp and time in seconds since the task began for each sample on a single row. We also record data identifying the start time (UTC), total_time in seconds to execute the task, and the total_number_of_errors in the first row of the saved datafile.
For example the row:
"""
line_number                                 2
x                                   57.681159
y                                   87.391304
correct_path                            2 ~ B
actual_path                             2 ~ B
UTC_Timestamp                  1752629732.528
seconds                                12.047
epoch_time_in_seconds_start               NaN
total_time                                NaN
total_number_of_errors                    NaN
is_error                                False
"""
tells us that the sampled point belonged to line number 2 (the second line drawn), the x, y coordinates were ~57.7, 87.4, the intended (correct) path was from the circle labeled '2' to the circle labeled 'B'. And, we can see that the line was correctly drawn from 2-->B. The point was collected at 12.047 seconds into the task, and that the line this point was part of was not in error. 
### Metadata:
The mapping between the behavioral data stored in the preprocessed/ files and the approprite user and 'trails' screen is embedded in the csv file name. The filenames take the form: <user_id>-hexcode-hexcode-hexcode-hexcode-hexcode-<trail_id>.csv. For example: M10992205-ad1a510e-82d8-4085-bdc3-e3567610e3b2-trail3.csv would indicate user_id M10992205's trail3 data.
### Task Stimuli Data:
The target stimuli for each trail screen are stored in the trail_making_config.json file. For each trail set, we store the radius of the circle, the fontsize, and the item features: order (the order in which the user should connect the circle), the center of the circle in x and y (cx, cy), and the label presented in the circle. We can then link these locations back to the behavioral data using the "actual_path" field. 
For example, the line that the point sample provided above could be localized to a line connecting a specific circle target location pair using the dictonary for trial 3:
"""
{'r': 17,
 'fontSize': 22,
 'items': [{'order': 1, 'cx': 150, 'cy': 200, 'label': '1'},
  {'order': 2, 'cx': 208, 'cy': 105, 'label': 'A'},
  {'order': 3, 'cx': 279, 'cy': 199, 'label': '2'},
  {'order': 4, 'cx': 214, 'cy': 152, 'label': 'B'},
  {'order': 5, 'cx': 214, 'cy': 246, 'label': '3'},
  {'order': 6, 'cx': 69, 'cy': 247, 'label': 'C'},
  {'order': 7, 'cx': 45, 'cy': 111, 'label': '4'},
  {'order': 8, 'cx': 127, 'cy': 141, 'label': 'D'}]}
"""
So the line from 2 -> B should connect circles of radius=17 between (x=208, y=105) and (x=214, y=152). Note that I am only using trail 3 as an example because it is short; we will not analyze trails 1 or 3. 

# The Analyses
We will generate code to perform some descriptive analyses of the data that will then be forwarded to additional analyses - e.g. relating to age, physiological factors, other behavioral metrics, etc.
## Basic Summary Statistics:
The data files already provide a time to complete measure for the entire trails 2 (TrailsA condition) and 4 (TrailsB condition) tasks. However, this time incldues errors as well as correct lines. As such, we will also need to generate gross time to completion for each of trails 2 (TrailsA condition) and 4 (TrailsB condition), removing the time spent drawing lines in error in TrailsA and TrailsB separately. We can report this as total_time_A, error_time_A, correct_time_A, total_time_B, error_time_B, correct_time_B. We will also want to report the total number of errors as error_count_A, error_count_B. 
## More Advanced Summary Statistics:
We would like to compute some additional summary statistics that take advantage of the digital nature of this version of the Trails A/B task. We will want to keep all statistics below separate for correct and error trials. 
### Correct Trials
*Ink Time vs. Think Time:* One interesting concept here is 'ink time' vs 'think time'. That is, how much time to users spend drawing lines (ink time) vs. deciding where to draw the next line, once they enter a circle (think time). We can compute these for each connected circle pair by as the time spent within the radius of the originating circle before (think time) and the time spent drawing the line from the edge of the originating circle to the edge of the destination circle (ink time). 
*Drawing Speed & Variability:* The average speed (pixels/second) of the line connecting the edges of the originating and destination circles - and the standard deviation of that speed. 
*Path Optimality:* For each circle pair, we can compute the minimum distance between the circles and compare that to the distance traveled by the user.
### Incorrect Trials
We should compute the metrics derived for correct trials on incorrect trials - when they are present. Many datasets will have no errors at all - or only on the more difficult TrailsB condition. 

# Data Outputs
The Basic Summary Statistics will be reported as the totals across the TrailsA and the totals across the TrailsB conditions separately.
The More Advanced Summary Statistics will be reported at two levels: 1. the average across the TrailsA and TrailsB separately - broken out by correct vs. error trials (if present), and 2. Item by item mean statistics for each line/circle. E.g. we would report the 'think time' spent in each circle, the drawing speed, variability, optimality, for each line - again broken out by correct and incorrect trials. The summary data in 1 will be saved as a single CSV for all Ss. The item-by-item data in 2 will be saved as one CSV file per user. 

# Data Conditioning Considerations
The data sampling while users are drawing between circle edges should be relatively consistent at ~60Hz. However, we may want to resample the line segments between circles to a uniform 60Hz timeline. The data sampling may vary significantly within circles, when the stylus is only minimally moving. However, this is not a concern since we really only want to know the amout of time spent between the stylus entering the circle and the stylus leaving the circle once again. 

# Visualizations
I would like to generate some compelling visualizations for individual and group data. 
## Individual data
Plot the circles described in the trail_making_config.json file, then overlay individual participant data with the lines drawn by the participant, color coded by speed. Shade the individual circles with a color reflecting the amount of 'think time' spent in each circle. 
## Group data
Plot the circles described in the trail_making_config.json file, then overlay group data reflecting the spatial distribution of the drawn paths between lines (e.g. as a shaded area with the color reflecting the average drawing speed) and the circles reflecting the average 'think time'. 
## Suggestions welcome
The above are rough ideas. I would like to hear thoughts on how we might enhance the visualizations. 