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
    In this test, you will be shown 50 tweets, one at a time. All tweets are related to recent news in the USA. In order to do the survey, you need to be
         familiar with the <a href = "https://en.wikipedia.org/wiki/Democratic_Party_presidential_debates,_2016" target="_blank">Democratic</a> and <a href = "https://en.wikipedia.org/wiki/Republican_Party_presidential_debates,_2016" target="_blank">Republican</a> parties. If you are not familiar with these parties, please click on the link above to access their Wikipedia descriptions.

    <br>
    <br>

Your task is to judge each tweet and answer three questions about it. For the third question you are also required to fill in a text box explaining your answer.
Please feel free to follow the URLs posted in the tweets if the tweet text does not provide enough information.


<br>

	<!--If a tweet is criticizing one democrat candidate to support another democrat candidate, then that should still be considered a pro-democratic tweet.
	As long as the intention of the tweet is to support some candidate from the democratic party in the presidential elections or the democratic party in general, it should be considered leaning pro-democratic.
	Similar reasoning should also be used for judging pro-republican tweets.-->

  <!--<i>Should we add some desription of each of these properties?</i>-->

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

<p> The task consists of 50 tweets, one per page. You need to give your judgment for all of them in order to complete the task. </p>

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




    list($tweet_id, $sname, $tweet_text, $tweet_link) = $tweet_obj;
    //list($tweet_id, $tweet_text) = $tweet;


    ?>






    <h1 id="header"><?= TITLE ?></h1>
    <div class="well clearfix">
    <div class="hero-unit">
    <!-- <form action="survey.php" method="get"> -->
    <?php
    /*$sname = "CNN";
    $tweet_text = "What makes a person presidential is wheter or not they can keep the peace.";
    $tweet_id = 853634729765216256;
    */
    ?>
    <?php
    if ($page <= $npages){
        $lastpage=0;
    ?>


        <?php
        if ($workerid % 2==1){
            $lastpage=0;
        ?>

            <p class="lead">
                <!--<b><em><a href="https://twitter.com/<?= $sname ?>"><?= $sname ?></a>: </em></b>-->
            <!--&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Author's bias : <?= round($pub,3); ?>
            <input class='pull-right' style='width: 20%;' type='range' value='<?= 50-50*$pub; ?>' disabled />
            <br><br>-->
            <p class="lead"><b><em><a href="https://twitter.com/<?= $sname ?>"><?= $sname ?></a>: </em></b>
            <?= linkify_text($tweet_text) ?><br>
            <!--<b><em><a href="https://twitter.com/<?= tweet_url($tweet_id, $sname) ?>"><?= "Tweet Link" ?></a> </em></b>-->
            <!--Promoter Bias : <?= round($prom,3); ?>-->
            <!--<br><br>
            Retweeters' political leaning:
            <input class='pull-right' style='width: 20%;' type='range' value='<?= 50-50*$prom; ?>' disabled />
            -->
            </p>
            <p class="tweet-text lead"></p>
            </div>
            <hr>
            <fieldset>
            <!--<legend>Q. What is the leaning of the tweet? </legend>-->

        <?php
        }
        if ($workerid % 2==0){
            $lastpage=0;
        ?>

            <p class="lead">
                <!--<b><em><a href="https://twitter.com/<?= $sname ?>"><?= $sname ?></a>: </em></b>-->
            <!--&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Author's bias : <?= round($pub,3); ?>
            <input class='pull-right' style='width: 20%;' type='range' value='<?= 50-50*$pub; ?>' disabled />
            <br><br>-->
            <p class="lead"><b><em>Tweet: </em></b>
            <?= linkify_text($tweet_text) ?><br>
            <!--<b><em><a href="https://twitter.com/<?= tweet_url($tweet_id, $sname) ?>"><?= "Tweet Link" ?></a> </em></b>-->
            <!--Promoter Bias : <?= round($prom,3); ?>-->
            <!--<br><br>
            Retweeters' political leaning:
            <input class='pull-right' style='width: 20%;' type='range' value='<?= 50-50*$prom; ?>' disabled />
            -->
            </p>
            <p class="tweet-text lead"></p>
            </div>
            <hr>
            <fieldset>
            <!--<legend>Q. What is the leaning of the tweet? </legend>-->


        <?php
        }
        $is_answered = check_asnwered($workerid, $tweet_id);




        if ($is_answered == 0){

            echo('<legend>Q1. Is the tweet likely to have a considerably different reaction from a democrat reader and a republican reader? </legend>');

            echo('<label class="radio">');
            echo('<input type="radio" name="a" id="a1" value="1" onClick="radio_btn_clicked(\'a1\')">');
            echo('<div id = "t1_text"> Yes </div>');
            echo('</label>');

            echo('<label class="radio">');
            echo('<input type="radio" name="a" id="a2" value="2" onClick="radio_btn_clicked(\'a2\')">');
            echo('<div id = "t2_text"> No </div>');
            echo('</label>');



            echo('<br><br><legend>Q2. Is the tweet an objective statement (irrespective of being true or not) that can be fact checked, or is it a subjective opinion?
                </legend>');

            echo('<label class="radio">');
            echo('<input type="radio" name="b" id="b1" value="1" onClick="radio_btn_clicked(\'b1\')">');
            echo('<div id = "t1_text"> Objective statement </div>');
            echo('</label>');

            echo('<label class="radio">');
            echo('<input type="radio" name="b" id="b2" value="2" onClick="radio_btn_clicked(\'b2\')">');
            echo('<div id = "t2_text"> Subjective opinion </div>');
            echo('</label>');

            echo('<br><br><legend>Q3. Would you share this tweet with your friends?  </legend>');

            echo('<label class="radio">');
            echo('<input type="radio" name="c" id="c1" value="1" onClick="radio_btn_clicked(\'c1\')">');
            echo('<div id = "t1_text"> Yes </div>');
            echo('</label>');

            echo('<label class="radio">');
            echo('<input type="radio" name="c" id="c2" value="2" onClick="radio_btn_clicked(\'c2\')">');
            echo('<div id = "t2_text"> No </div>');
            echo('</label>');

            echo('<br> Explain the reasons for your choice </legend>');

            echo('<label class="text">');
            //echo('<textarea rows="4" cols"50" nam="test" required>'</textarea>');
            echo('<textarea name="text" cols="20" rows="8" id="text_1" wrap onkeypress="return onTestChange();" ></textarea>');

//            echo('<input type="textarea" name="text" id="text_1" rows="14" cols="500" value="" size="200" >');
            //echo('<input type="textarea" name="text" id="text_1" style="width:500px; height:120px;" value="" wrap >');
            echo('</label>');


                //            $same_page = sprintf("survey.php?p=%s&w=%s", $page, $workerid)
            ?>

            <?php

/*            echo('<br><br><legend> Explain the reasons for your choice. </legend>');

            echo('<label class="radio">');
            echo('<input type="radio" name="d" id="d1" value="1" onClick="radio_btn_clicked(\'d1\')">');
            echo('<div id = "t1_text"> Factual </div>');
            echo('</label>');

            echo('<label class="radio">');
            echo('<input type="radio" name="d" id="d2" value="2" onClick="radio_btn_clicked(\'d2\')">');
            echo('<div id = "t2_text"> Opinion </div>');
            echo('</label>');*/


        } else {


            list($given_answer_a, $given_answer_b, $given_answer_c, $given_answer_text) = get_answer($workerid, $tweet_id,$lastpage);

            $given_answer_a = (int) $given_answer_a;
            $given_answer_b = (int) $given_answer_b;
            $given_answer_c = (int) $given_answer_c;
            $given_answer_text = (string) $given_answer_text;
            $reason_share = $given_answer_text;


            echo('<legend>Q1. Is the tweet likely to have a considerably different reaction from a democrat reader and a republican reader? </legend>');
            echo('<label class="radio">');
            if ($given_answer_a == 1){
                echo('<input type="radio" name="a" id="a1" value="1" checked disabled>');
            } else {
                echo('<input type="radio" name="a" id="a1" value="1" disabled>');
            }
            echo('<div id = "t1_text"> Yes </div>');
            echo('</label>');


            echo('<label class="radio">');
            if ($given_answer_a == 2){
                echo('<input type="radio" name="a" id="a2" value="2" checked disabled>');

            } else {
                echo('<input type="radio" name="a" id="a2" value="2" disabled>');
            }
            echo('<div id = "t2_text"> No </div>');
            echo('</label>');





            echo('<br><br><legend>Q2. Is the tweet an objective statement (irrespective of being true or not) that can be fact checked, or is it a subjective opinion? </legend>');
            echo('<label class="radio">');
            if ($given_answer_b == 1){
                echo('<input type="radio" name="b" id="b1" value="1" checked disabled>');
            } else {
                echo('<input type="radio" name="b" id="b1" value="1" disabled>');
            }
            echo('<div id = "t1_text"> Objective statement </div>');
            echo('</label>');


            echo('<label class="radio">');
            if ($given_answer_b == 2){
                echo('<input type="radio" name="b" id="b2" value="2" checked disabled>');

            } else {
                echo('<input type="radio" name="b" id="b2" value="2" disabled>');
            }
            echo('<div id = "t2_text"> Subjective opinion </div>');
            echo('</label>');






            echo('<br><br><legend>Q3. Would you share this tweet with your friends? Explain the reasons for your choice. </legend>');
            echo('<label class="radio">');
            if ($given_answer_c == 1){
                echo('<input type="radio" name="c" id="c1" value="1" checked disabled>');
            } else {
                echo('<input type="radio" name="c" id="c1" value="1" disabled>');
            }
            echo('<div id = "t1_text"> Yes </div>');
            echo('</label>');


            echo('<label class="radio">');
            if ($given_answer_c == 2){
                echo('<input type="radio" name="c" id="c2" value="2" checked disabled>');

            } else {
                echo('<input type="radio" name="c" id="c2" value="2" disabled>');
            }
            echo('<div id = "t2_text"> No </div>');
            echo('</label>');



            /*
            echo('<br><br><legend>Q4. Do you find the above tweet factual or opinion? </legend>');
            echo('<label class="radio">');
            if ($given_answer_d == 1){
                echo('<input type="radio" name="d" id="d1" value="1" checked disabled>');
            } else {
                echo('<input type="radio" name="d" id="d1" value="1" disabled>');
            }
            echo('<div id = "t1_text"> Factual </div>');
            echo('</label>');


            echo('<label class="radio">');
            if ($given_answer_d == 2){
                echo('<input type="radio" name="d" id="d2" value="2" checked disabled>');

            } else {
                echo('<input type="radio" name="d" id="d2" value="2" disabled>');
            }
            echo('<div id = "t2_text"> Opinion </div>');
            echo('</label>');
            */


        }

    }
    # end of if condition for "if its a survey question"
            #else{
    if ($page == $npages+1){
        $lastpage=1;
        //print("salam");
    ?>
        <p class="lead">Almost there ...</p>
        <p class="lead"><b>To help us interpret the survey better, please tell us how would you identify your own political affiliation.</b></p>
		<p> (This information will not be made public and would only be used for academic research purposes.) </p>
        <!-- <p class="lead"><b><em>Tweet: </em></b><?= linkify_text($tweet_text) ?></p> -->
        <p class="tweet-text lead"></p>
        </div>

        <fieldset>
        <legend>Q. My political affiliation is: </legend>

    <?php

    $is_answered = check_asnwered($workerid, $tweet_id);


    if ($is_answered == 0){
        echo('<label class="radio">');
        echo('<input type="radio" name="a" id="a1" value="1" onClick="radio_btn_clicked(\'a1\')">');
        echo('<div id = "t1_text"> Democratic </div>');
        echo('</label>');

        echo('<label class="radio">');
        echo('<input type="radio" name="a" id="a2" value="2" onClick="radio_btn_clicked(\'a2\')">');
        echo('<div id = "t2_text"> Republican </div>');
        echo('</label>');

        echo('<label class="radio">');
        echo('<input type="radio" name="a" id="a3" value="3" onClick="radio_btn_clicked(\'a3\')">');
        echo('<div id = "t3_text"> Neither of the two </div>');
        echo('</label>');

        echo('<label class="radio">');
        echo('<input type="radio" name="a" id="a4" value="4" onClick="radio_btn_clicked(\'a4\')">');
        echo('<div id = "t3_text"> Dont want to disclose </div>');
        echo('</label>');





    } else {


        list($given_answer_a, $given_answer_b, $given_answer_c, $given_answer_d) = get_answer($workerid, $tweet_id,$lastpage);

        $given_answer_a = (int) $given_answer_a;
        $given_answer_b = (int) $given_answer_b;
        $given_answer_c = (int) $given_answer_c;
        $given_answer_d = (int) $given_answer_d;


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


        echo('<label class="radio">');
        if ($given_answer_a == 3){
            echo('<input type="radio" name="a" id="a3" value="3" checked disabled>');

        } else {
            echo('<input type="radio" name="a" id="a3" value="3" disabled>');
        }
        echo('<div id = "t3_text"> Neither of the two </div>');
        echo('</label>');


        echo('<label class="radio">');
        if ($given_answer_a == 4){
            echo('<input type="radio" name="a" id="a4" value="4" checked disabled>');

        } else {
            echo('<input type="radio" name="a" id="a4" value="4" disabled>');
        }
        echo('<div id = "t3_text"> Dont want to disclose </div>');
        echo('</label>');





        }

    }

    ?>
    </fieldset>


    <hr>

    <?php
        $page_next = $page + 1;

        //$is_dem_tweet = 1; //means its a dem tweet
        //if ($side == 'REP'){
        //    $is_dem_tweet = 0; //means its a rep tweet
        //}

        //var_dump("here");
        //var_dump($page_next);

        if ($page <= $npages){
        $lastpage=0;
        echo '<button  id = "btn_submit" class="btn" onClick="sendQueryWithAnswer('. $page_next .','. $workerid . ');" disabled>Continue to next Tweet</button>';

        }
        else{
            $lastpage=1;
            echo '<button  id = "btn_submit" class="btn" onClick="sendQueryWithAnswerLast('. $page_next .','. $workerid . ');" disabled>Continue to next Tweet</button>';
        }

    ?>
    <!-- </form> -->

    <p class="lead pull-right"><?= $page ?> / <?= $npages ?></p>
    </div>

    <?php
} // End function judge_test

// Display the code and final content
//function finish($workerid, $total_responses, $total_correct_responses, $total_responses_test, $total_correct_responses_test) {
function finish($workerid, $total_responses) {

    $passcode = "twitter app survey crowd signal" . $workerid;
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
$rc       = (int) get_default($_GET, "rc", "0");
$rd       = (int) get_default($_GET, "rd", "0");
$text       = (string) get_default($_GET, "text", "");

// If we dont have a workerid then this is a new user
if (!($workerid > 0)) {
    $workerid = mturk_next_id();
}

// Log page access
mturk_access_log($workerid);

// Get the current tweet set

$num_sets = 1;
$set_number = $workerid % $num_sets;
$set = survey_data_set($set_number);
//var_dump(count($set));

$npages = count($set);
//print($npages);
//print("\n");
//print($page);
$npages_test = count($set);

// Page 0 : start page
// Page 1 - count(set) : the tweet pages
// Page count(set) + 1 : the finish page

//FIXLATER
if (!(0 <= $page and $page <= $npages + 2)) {
    $page = 0;
}


// Reponses are vaild when the page is in the following range
if (2 <= $page and $page <= $npages + 2) {
    // Get the detils of last page

    if($page <= $npages+1){
        list($tweet_id, $sname, $tweet_text, $tweet_link) = $set[$page - 2];
    }
    if($page == $npages +2){
        list($tweet_id, $sname, $tweet_text, $tweet_link) = $set[$page - 3];
        $tweet_id=1;
    }
    //list($tweet_id, $tweet_text) = $tweet;


    // Save response
	//Check here if the $ra variable is still 0, then redirect to the same page with an error message

    //survey_response($workerid, $tweet1_id, $tweet2_id, $ra, $answer, $issue, $type)
    //$answer = 0;
    //if ($side == "DEM"){
    //    $answer = 1;
    //} else {
    //    $answer = 2;
    //}


    $is_answered = check_asnwered($workerid, $tweet_id);
    if ($is_answered == 0){
        survey_response($workerid, $tweet_id, $ra, $rb, $rc, $rd, $text);

    }
}


// Main page logic
include("_header.php");

if ($page == 0) {
    start($workerid, $npages);
}
else if ($page <= $npages+1) {
    if($page <= $npages){
        $tweet_obj = $set[$page-1];
    }
    else{
        $tweet_obj = [$tweet_id+1, $sname, $tweet_text, $tweet_link];
    }
    judge($workerid, $page, $npages, $tweet_obj);
    //var_dump($page);
    //var_dump($npages);
}
else {
    //var_dump("in last");
    list($total_responses) = survey_count_response($workerid);
	//list($total_responses, $total_correct_responses) = survey_count_response($workerid);
    //list($total_responses_test, $total_correct_responses_test) = survey_count_response_test($workerid);
    if ($total_responses == count($set)+1) {
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



function onTestChange() {
    var key = window.event.keyCode;

    // If the user has pressed enter
    if (key === 13) {
        document.getElementById("text_1").value = document.getElementById("text_1").value + "\n ";
        return false;
    }
    else {
        return true;
    }
}

function sendQueryWithAnswer(p, w)
{

    ra = 0;
    rb = 0;
    rc = 0;
    rd = 0;
    text="";

    if (document.getElementById("a1").checked){
        ra = 1;
    }

    if (document.getElementById("a2").checked){
        ra = 2;
    }



    if (document.getElementById("b1").checked){
         rb = 1;
    }

    if (document.getElementById("b2").checked){
         rb = 2;
    }


    if (document.getElementById("c1").checked){
        rc = 1;
    }

    if (document.getElementById("c2").checked){
        rc = 2;
    }

    //if (document.getElementById("text_1").checked){
    text = document.getElementById("text_1").value;
        //text = "checked";
    //}


    callNext(p, w, ra, rb, rc, rd,text);

    
    
}


function sendQueryWithAnswerLast(p, w)
{
    
    ra = 0;
    rb = 0;
    rc = 0;
    rd = 0;
    
    
    if (document.getElementById("a1").checked){
        ra = 1;
    }
    
    if (document.getElementById("a2").checked){
        ra = 2;
    }
    
    if (document.getElementById("a3").checked){
        ra = 3;
    }
    
    if (document.getElementById("a4").checked){
        ra = 4;
    }
    
    
    
    callNext(p, w, ra, rb, rc, rd);
    
    
    
}

function radio_btn_clicked(btn_id)
{
    
    
    document.getElementById("btn_submit").disabled = false;
    // if (btn_id == "a1" || btn_id == "a2" || btn_id == "a3"){
    //     document.getElementById("b1").disabled = false;
    //     document.getElementById("b2").disabled = false;
    //     document.getElementById("b3").disabled = false;
    // }
    
    //if (btn_id == "a1" || btn_id == "a2" || btn_id == "a3" || btn_id == "a4" ){
    //if ((btn_id == "a1" || btn_id == "a2") ){
    // if (btn_id == "a1" || btn_id == "a2"){
    //    document.getElementById("btn_submit").disabled = false;

    //}
}
</script>

</html>
