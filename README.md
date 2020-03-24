# perception-bias-fake-news
Perception Project
This project includes different directory files such as AMT_survey, Code, data-set.

AMT_survey:
Here you can find the code that we created the survey then set up it in Amazon Mechanical Turk to collect the data.
This project results are reported mainly based on sp_claims_gt1, sp_claims_gt2, sp_claims_gt3 surveys.

To ensure that our test design was robust to variations in deployment and to common survey biases, 
we evaluated the effect of the survey variations by testin them that includes: 

Sample Effects: The survey variations we compare in the context of sample effects are: – Running our test using MTurk Masters vs. naive MTurkers.
– Running our test using a census-representative sample of participants recruited by Survey Sampling International(SSI participants) vs. MTurk Masters(experts participants).

Answer Choice Effects: We evaluate the answer choice effects by comparing survey variations with 6- and 7-point scales with Snopes’ 5-point scale.

Satisficing and Incentive Effects: Satisficing is a commonly observed survey response effect in which respondents select what they consider the minimum acceptable answer, 
without fully considering their true feelings. 
Our truth perception test style surveys may be at a particular risk of satisficing because they encourage quick responses. 
Thus, we explore the effect of incentivizing participants for correct answers to evaluate whether satisficing may be affecting our test.

Thus to test them we set up other sruveys that you can find the related survey code there.


data-set:
After we collected the answeres from AMT workers we collected them, preprocess and prepare them to do some analysis.

Amt- workers- features.csv):
In each row, we have the worker id and the value of their demographic features as follows:
 Worker_id, age, degree, employment, gender, income, marital_status, political_view, race
 
Workers- perception- 1.csv:
Since asking one worker to label 150 claims was not possible, we divided our 150 news stories to three groups that each of them includes
50 claims (10 False, 10 Mostly False, 10 Mixture, 10 Mostly True, and 10 True). Then for each group, 100 AMT workers were asked to label them.
Includes first group of 50 news stories with the following header:
Text-id: unique id assigned to each news story.
Text: context of the news story.
Ground-truth: the ground truth label that given by Snopes (False, Mostly False, Mixture, Mostly True, True).
Worker-id: unique id assigned to each worker, and in the corresponding column, you can see how worker labels each news story.
We mapped these labels on a scale between 1 and 5. This means False, Mostly False, Mixture, Mostly True, and True mapped to 1, 2, 3, 4,and 5 respectively.

Workers- perception- 2.csv:
Same format as Workers-perception-1.csv, however, it includes the second group of 50 news stories.

Workers-perception-3.csv:
Same format as Workers-perception-1.csv, however, it includes the third group of 50 news stories.

Code:
Using analysis_fake-news_new_gt.py we preprocess the raw data collected from AMT workers. 
Then we extract the features from them and create the csv files that we explained them above.
Finally using predict_fake_news.py we tested different approaches for prediction.
 
 
