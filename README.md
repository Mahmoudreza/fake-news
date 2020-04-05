# Users' perception biases in judging (fake) news


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
We release the data set very soon!

Code:
Using analysis_fake-news_new_gt.py we preprocess the raw data collected from AMT workers. 
Then we extract the features from them and create the csv files that we explained them above.
Finally using predict_fake_news.py we tested different approaches for prediction.
 
 
