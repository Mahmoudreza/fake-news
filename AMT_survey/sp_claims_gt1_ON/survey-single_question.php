<?php
    include("_init.php");
    include("_utilities.php");
    
    const TITLE = "Analyzing Quality of Tweets";
    
    //Arrays for rendering the survey issue and questions
    
    
    
    
    // Show the mturk landing content
    function start($workerid, $npages) {
        
        $next_uri = sprintf("survey.php?p=1&w=%s", $workerid);
    ?>

<h1 id="header"><?= TITLE ?></h1>

<div id="mturk-intro" class="well">

<!-- <h4 style="color:red">Note: If you have done a HIT with us having the same name before, please do not attempt this one -- if you do, we will need to discard your answers and you won't get paid. </h4> -->
<h2>Welcome</h2>

<p>Hi there!
<br>
Thanks for taking the time to give us your valuable feedback!
<br>
We are a team of researchers from <a href = "http://socialnetworks.mpi-sws.org/" target="_blank"> Social Computing Research Group </a> at <a href = "http://mpi-sws.org" target="_blank"> Max Planck Institute for Software Systems</a>.

</p>
<p>
In this test, you will be shown 90 tweets, one at a time. All tweets are related to the <a href = "https://en.wikipedia.org/wiki/Democratic_Party_presidential_debates,_2016" target="_blank">Democratic</a> and <a href = "https://en.wikipedia.org/wiki/Republican_Party_presidential_debates,_2016" target="_blank">Republican</a> Party Presidential Debates 2015. If you are not familiar with these events, please click on the link above to access the Wikipedia description of the events.
<br>
<br>
Your task is to judge each tweet and answer whether or not you find the tweet to be (i) <b><i> informative</b></i>, (ii) <b><i>interesting</b></i>, (iii) <b><i>credible</b></i>, and (iv) <b><i>factual</b></i>.

Please feel free to follow the URLs posted in the tweets if the tweet text does not provide enough information.
<br>
<br>
<!--If a tweet is criticizing one democrat candidate to support another democrat candidate, then that should still be considered a pro-democratic tweet.
As long as the intention of the tweet is to support some candidate from the democratic party in the presidential elections or the democratic party in general, it should be considered leaning pro-democratic.
Similar reasoning should also be used for judging pro-republican tweets.-->

<i>Should we add some desription of each of these properties?</i>

</p>
<!--
<p>
For example, consider following three tweets related to the event of <b>US government shutdown of 2013</b>:
<br><br>
<b>Tweet 1: </b> <?php echo(linkify_text("I hate #Obamacare & I hate what #shutdown is doing to families across VA. Here’s my plan to address both: http://t.co/fUVlcwVuZ0")); ?>
<br>
<b>Tweet 2: </b> <?php echo(linkify_text("Is it just me...OR did you notice how much the #GOP #Congressional #Shutdown is reminiscent of Somalia Pirate's \"HOSTAGE TAKING\"! #GOP SCAM")); ?>
<br>
<br>


In this case, you would have to identify Tweet 1 as being posted by a Republican-leaning user and Tweet 2 as being posted by a Democratic-leaning user.

</p> -->


<p> </p>

<p> The task consists of 90 tweets, one per page. You need to give your judgment for all of them in order to complete the task. </p>

<p>

<!-- We have also included 5 <i>test</i> tweets at different points in the survey. These are tweets for which we know their type ('impartial' or 'not impartial'). At the end of the survey, we will show you for how many test questions you were able to guess the correct type. -->

<!-- <b> Earn a bonus! </b>

Just to make things a little more interesting, we have included 5 <i>test</i> tweets at different points in the survey. These are tweets for which we know their polarity (Democratic or Republican), and if you indicate the correct answer for all 5 of them you will be rewarded a <b>5$ bonus!</b>

<p>
-->
<!--
<b style = "color:red" >Note: </b> Due to the nature of the survey, this specific HIT can only be attempted once by a single AMT worker. So, if you have completed this survey once, please do not attempt it again. Thanks!
-->

</p>
</div>

<ul class="pager">
<li class="next">
<a href="<?= $next_uri ?>">Click here to start the test</a>
</li>
</ul>

<?php
    } // End function start
    // Do the actual judgement
    
    // Do the actual judgement
    function judge($workerid, $page, $npages, $tweet_obj) {
        // var_dump($page);
        // var_dump($npages);
        
        
        
        
        list($type, $topic, $side, $tweet) = $tweet_obj;
        list($tweet_id, $tweet_text) = $tweet;
        
        
        ?>






<h1 id="header"><?= TITLE ?></h1>
<div class="well clearfix">
<div class="hero-unit">
<!-- <form action="survey.php" method="get"> -->


<?php
    if ($page <= $npages-1){
        ?>





<p class="lead"><b><em><a href="https://twitter.com/<?= $type ?>"><?= $type ?></a>: </em></b><?= linkify_text($tweet_text) ?></p>
<p class="tweet-text lead"></p>
</div>
<hr>
<fieldset>
<legend>Q. What is the leaning of the tweet? </legend>

<?php
    
    $is_answered = check_asnwered($workerid, $tweet_id);
    
    
    if ($is_answered == 0){
        echo('<label class="radio">');
        echo('<input type="radio" name="a" id="a1" value="1" onClick="radio_btn_clicked(\'a1\')">');
        echo('<div id = "t1_text"> Pro Democratic </div>');
        echo('</label>');
        
        echo('<label class="radio">');
        echo('<input type="radio" name="a" id="a2" value="2" onClick="radio_btn_clicked(\'a2\')">');
        echo('<div id = "t2_text"> Pro Republican </div>');
        echo('</label>');
        
        
    } else {
        
        
        list($given_answer_a, $given_answer_b, $actual_answer) = get_answer($workerid, $tweet_id);
        
        $given_answer_a = (int) $given_answer_a;
        $given_answer_b = (int) $given_answer_b;
        $actual_answer = (int) $actual_answer;
        
        
        echo('<label class="radio">');
        if ($given_answer_a == 1){
            echo('<input type="radio" name="a" id="a1" value="1" checked disabled>');
        } else {
            echo('<input type="radio" name="a" id="a1" value="1" disabled>');
        }
        echo('<div id = "t1_text"> Pro Democratic </div>');
        echo('</label>');
        
        
        echo('<label class="radio">');
        if ($given_answer_a == 2){
            echo('<input type="radio" name="a" id="a2" value="2" checked disabled>');
            
        } else {
            echo('<input type="radio" name="a" id="a2" value="2" disabled>');
        }
        echo('<div id = "t2_text"> Pro Republican </div>');
        echo('</label>');
        
        
        
        
    }
    
    }
    # end of if condition for "if its a survey question"
    else{
        ?>
<p class="lead">Almost there ...</p>
<p class="lead"><b>To help us interpret the survey better, please tell us how would you identify your own political affiliation.</b></p>
<p> (This information will not be made public and would only be used for academic research purposes.) </p>
<!-- <p class="lead"><b><em>Tweet: </em></b><?= linkify_text($tweet_text) ?></p> -->
<p class="tweet-text lead"></p>
</div>

<fieldset>
<legend>Q. What do you identify yourself more as: </legend>

<?php
    
    $is_answered = check_asnwered($workerid, $tweet_id);
    
    
    if ($is_answered == 0){
        echo('<label class="radio">');
        echo('<input type="radio" name="a" id="a1" value="1" onClick="radio_btn_clicked(\'a1\')">');
        echo('<div id = "t1_text"> Democratic-leaning </div>');
        echo('</label>');
        
        echo('<label class="radio">');
        echo('<input type="radio" name="a" id="a2" value="2" onClick="radio_btn_clicked(\'a2\')">');
        echo('<div id = "t2_text"> Republican-leaning </div>');
        echo('</label>');
        
        
        
    } else {
        
        
        list($given_answer_a, $given_answer_b, $actual_answer) = get_answer($workerid, $tweet_id);
        
        $given_answer_a = (int) $given_answer_a;
        $given_answer_b = (int) $given_answer_b;
        $actual_answer = (int) $actual_answer;
        
        
        echo('<label class="radio">');
        if ($given_answer_a == 1){
            echo('<input type="radio" name="a" id="a1" value="1" checked disabled>');
        } else {
            echo('<input type="radio" name="a" id="a1" value="1" disabled>');
        }
        echo('<div id = "t1_text"> Democratic</div>');
        echo('</label>');
        
        
        echo('<label class="radio">');
        if ($given_answer_a == 2){
            echo('<input type="radio" name="a" id="a2" value="2" checked disabled>');
            
        } else {
            echo('<input type="radio" name="a" id="a2" value="2" disabled>');
        }
        echo('<div id = "t2_text"> Republican</div>');
        echo('</label>');
        
        
        
        
        
        
    }
    
    }
    
    ?>
</fieldset>


<hr>

<?php
    $page_next = $page + 1;
    
    $is_dem_tweet = 1; //means its a dem tweet
    if ($side == 'REP'){
        $is_dem_tweet = 0; //means its a rep tweet
    }
    
    echo '<button  id = "btn_submit" class="btn" onClick="sendQueryWithAnswer('. $page_next .','. $workerid . ',' . $is_dem_tweet . ');" disabled>Continue to next Tweet</button>';
    
    ?>
<!-- </form> -->

<p class="lead pull-right"><?= $page ?> / <?= $npages ?></p>
</div>

<?php
    } // End function judge_test
    
    // Display the code and final content
    //function finish($workerid, $total_responses, $total_correct_responses, $total_responses_test, $total_correct_responses_test) {
    function finish($workerid, $total_responses) {
        
        $passcode = "twitter app survey search engine bias" . $workerid;
        $passcode = sha1($passcode);
    ?>

<h1 id="header"><?= TITLE ?></h1>

<div class="well">
<center>
<h2>Thank You!</h2>
<p class="lead">
Please submit the following code to AMT for collecting your
remuneration.
</p>

<p class="lead"><?= $passcode ?></p>
</center>
</div>

<!-- <div class="well">
<center>
<h3>Responses for test tweets</h3>

<?php echo("<p class=\"lead\" > <b> You solved ". $total_correct_responses_test . " out of ". $total_responses_test. " test problems correctly! </b></p>");
    
    // if ($total_correct_responses_test == $total_responses_test){
    //     echo("<p class=\"lead\" > <b> Your bonus reward will be transferred to your account shortly. </b></p>");
    // }
    ?>
</center>
</div>
-->
<?php
    } // End function finish
    
    // Display the code and final content
    function finish_error() {
    ?>

<h1 id="header"><?= TITLE ?></h1>

<div class="well">
<center>
<h2>Something went wrong!</h2>

<p class="lead">
We don't have all the responses from you for this test.<br>
We have noted the error and informed our admins.
</p>
</center>
</div>

<?php
    } // End function finish_error
    
    // Get the parameters
    $page     = (int) get_default($_GET, "p", "0");
    $workerid = (int) get_default($_GET, "w", "0");
    $ra       = (int) get_default($_GET, "ra", "0");
    $rb       = (int) get_default($_GET, "rb", "0");
    
    // If we dont have a workerid then this is a new user
    if (!($workerid > 0)) {
        $workerid = mturk_next_id();
    }
    
    // Log page access
    mturk_access_log($workerid);
    
    // Get the current tweet set
    
    $num_sets = 4;
    $set_number = $workerid % $num_sets;
    $set = survey_data_set($set_number);
    //var_dump(count($set));
    
    $npages = count($set);
    $npages_test = count($set);
    
    // Page 0 : start page
    // Page 1 - count(set) : the tweet pages
    // Page count(set) + 1 : the finish page
    
    //FIXLATER
    if (!(0 <= $page and $page <= $npages + 1)) {
        $page = 0;
    }
    
    
    // Reponses are vaild when the page is in the following range
    if (2 <= $page and $page <= count($set) + 1) {
        // Get the detils of last page
        
        list($type, $topic, $side, $tweet) = $set[$page - 2];
        list($tweet_id, $tweet_text) = $tweet;
        
        
        // Save response
        //Check here if the $ra variable is still 0, then redirect to the same page with an error message
        
        //survey_response($workerid, $tweet1_id, $tweet2_id, $ra, $answer, $issue, $type)
        $answer = 0;
        if ($side == "DEM"){
            $answer = 1;
        } else {
            $answer = 2;
        }
        
        
        $is_answered = check_asnwered($workerid, $tweet_id);
        if ($is_answered == 0){
            survey_response($workerid, $tweet_id, $ra, $rb, $answer, $topic, $type);
            
        }
    }
    
    
    // Main page logic
    include("_header.php");
    
    if ($page == 0) {
        start($workerid, $npages);
    }
    else if ($page <= count($set)) {
        
        
        //["Shutdown", "Dem", [382980870954835969, "Republicans want to shut down the government just to go back to the days when health insurance companies were in charge. #getcovered", "getcovered", 0.51662043074926434], [386587493140680705, "Police Remove Vietnam War Veterans at Memorial Wall http://t.co/MZb1KZXuNR #makedclisten #dem shutdown #obamashutdown", "obamashutdown", 0.6167330694968004]]
        
        $tweet_obj = $set[$page-1];
        judge($workerid, $page, $npages, $tweet_obj);
    }
    else {
        list($total_responses) = survey_count_response($workerid);
        //list($total_responses, $total_correct_responses) = survey_count_response($workerid);
        //list($total_responses_test, $total_correct_responses_test) = survey_count_response_test($workerid);
        if ($total_responses == count($set)) {
            finish($workerid, $total_responses);
            //finish($workerid, $total_responses, $total_correct_responses, $total_responses_test, $total_correct_responses_test);
        }
        else {
            finish_error();
        }
    }
    ?>

</div> <!-- root container -->
</body>

<script type="text/javascript">

function sendQueryWithAnswer(p, w, is_dem_tweet)
{
    
    ra = 0;
    rb = 0;
    
    
    if (document.getElementById("a1").checked){
        ra = 1;
    }
    
    if (document.getElementById("a2").checked){
        ra = 2;
    }
    
    //if (document.getElementById("a3").checked){
    //    ra = 3;
    //}
    
    //if (document.getElementById("a4").checked){
    //    ra = 4;
    //}
    
    
    // if (document.getElementById("b1").checked){
    //     rb = 1;
    // }
    
    // if (document.getElementById("b2").checked){
    //     rb = 2;
    // }
    
    // if (document.getElementById("b3").checked){
    //     rb = 3;
    // }
    
    
    callNext(p,w,ra,rb);
    
    
    
}


function radio_btn_clicked(btn_id)
{
    // if (btn_id == "a1" || btn_id == "a2" || btn_id == "a3"){
    //     document.getElementById("b1").disabled = false;
    //     document.getElementById("b2").disabled = false;
    //     document.getElementById("b3").disabled = false;
    // }
    
    //if (btn_id == "a1" || btn_id == "a2" || btn_id == "a3" || btn_id == "a4" ){
    if (btn_id == "a1" || btn_id == "a2" ){
        // if (btn_id == "a1" || btn_id == "a2"){
        document.getElementById("btn_submit").disabled = false;
        
    }
}
</script>

</html>
