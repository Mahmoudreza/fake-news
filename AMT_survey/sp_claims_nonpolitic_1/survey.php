<?php
include("_init.php");
include("_utilities.php");

    const TITLE = "Analyzing claims on social media";

//Arrays for rendering the survey issue and questions


// Show the mturk landing content
function start($workerid, $npages) {

    $next_uri = sprintf("survey.php?p=1&w=%s", $workerid);

?>


<h1 id="header"><?= TITLE ?></h1>

<div id="mturk-intro" class="well">

    <h2>Welcome</h2>

    <p>Hi there!
    <br>
    Thanks for taking the time to give us your valuable feedback!
    <br><br>
    We are a team of researchers from <a href = "http://socialnetworks.mpi-sws.org/" target="_blank"> Social Computing Research Group </a> at <a href = "http://mpi-sws.org" target="_blank"> Max Planck Institute for Software Systems</a>.

    </p>
    <p>
In this survey, you will be shown 100 claims, which are spreading or have spread over social media. We would like to evaluate the "truth value" of each claim based on your judgment. Your task is to read each claim and label it as either `I can confirm it to be false’, ‘Very likely to be false’, ‘Possibly false, ‘Possibly true', ‘Very likely to be true’, or ‘I can confirm it to be true’. 

<h3>Please do not conduct any web search or use any online/offline resources for verifying or validating the claim presented to you. Please use your best judgment (your instinctive gut based guess within a few seconds) to label the claims.</h3>



    </p>


<p> </p>

<p> The task consists of 100 claims, one per page. You need to give your judgment for all of them in order to complete the task. </p>

<p>


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
    if ($page <= $npages){
        $lastpage=0;
    ?>
        <b style = "color:black" >Claim: </b>

        <?php
        if ($workerid % 2==1){
            $lastpage=0;
        ?>

            <p class="lead">
                <!--<b><em><a href="https://twitter.com/<?= $sname ?>"><?= $sname ?></a>: </em></b>-->
            <!--&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Author's bias : <?= round($pub,3); ?>
            <input class='pull-right' style='width: 20%;' type='range' value='<?= 50-50*$pub; ?>' disabled />
            <br><br>-->
            <p class="lead">
            <?= linkify_text($tweet_text) ?>
            <!--<?= linkify_text($tweet_link) ?><br>-->
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

            <!--<p class="lead">-->
            <!--<p class="lead"><b><em>News: </em></b>-->
            <!--<?= linkify_text($tweet_text) ?><br>-->
            <!--<b><em><a href="https://twitter.com/<?= tweet_url($tweet_id, $sname) ?>"><?= "Tweet Link" ?></a> </em></b>-->
            <!--</p>-->
            <!--<p class="tweet-text lead"></p>-->
            <!--</div>-->
            <!--<hr>-->
            <!--<fieldset>-->

            <p class="lead">
            <p class="lead">
            <?= linkify_text($tweet_text) ?>
            <!--<?= linkify_text($tweet_link) ?><br>-->
            <!--<b><em><a href="https://twitter.com/<?= tweet_url($tweet_id, $sname) ?>"><?= "Tweet Link" ?></a> </em></b>-->
            </p>
            <p class="tweet-text lead"></p>
            </div>
            <hr>
            <fieldset>




        <?php
        }
        $is_answered = check_asnwered($workerid, $tweet_id);


        if ($is_answered == 0){

            echo('<legend>Please give your judgment about this claim.
                </legend>');
            echo('<label class="radio">');
            echo('<input type="radio" name="a" id="a1" value="1" onClick="radio_btn_clicked(\'a1\')">');
            echo('<div id = "t1_text"> I can confirm it to be false. </div>');
            echo('</label>');

            echo('<label class="radio">');
            echo('<input type="radio" name="a" id="a2" value="2" onClick="radio_btn_clicked(\'a2\')">');
            echo('<div id = "t2_text"> Very likely to be false. </div>');
            echo('</label>');


            echo('<label class="radio">');
            echo('<input type="radio" name="a" id="a3" value="3" onClick="radio_btn_clicked(\'a2\')">');
            echo('<div id = "t3_text">Possibly false.</div>');
            echo('</label>');

            echo('<label class="radio">');
            echo('<input type="radio" name="a" id="a4" value="4" onClick="radio_btn_clicked(\'a2\')">');
            echo('<div id = "t4_text"> Can\'t tell.</div>');
            echo('</label>');

            echo('<label class="radio">');
            echo('<input type="radio" name="a" id="a5" value="5" onClick="radio_btn_clicked(\'a2\')">');
            echo('<div id = "t5_text">Possibly true. </div>');
            echo('</label>');



            echo('<label class="radio">');
            echo('<input type="radio" name="a" id="a6" value="6" onClick="radio_btn_clicked(\'a2\')">');
            echo('<div id = "t6_text">Very likely to be true.</div>');
            echo('</label>');


            echo('<label class="radio">');
            echo('<input type="radio" name="a" id="a7" value="7" onClick="radio_btn_clicked(\'a2\')">');
            echo('<div id = "t7_text">I can confirm it to be true.</div>');
            echo('</label>');


            ?>

            <?php



        } else {


            list($given_answer_a, $given_answer_b, $given_answer_c, $given_answer_text) = get_answer($workerid, $tweet_id,$lastpage);

            $given_answer_a = (int) $given_answer_a;
            $given_answer_b = (int) $given_answer_b;
            $given_answer_c = (int) $given_answer_c;
            $given_answer_text = (string) $given_answer_text;
            $reason_share = $given_answer_text;


            echo('<legend>Please give your judgment about this claim. </legend>');
            echo('<label class="radio">');
            if ($given_answer_a == 1){
                echo('<input type="radio" name="a" id="a1" value="1" checked disabled>');
            } else {
                echo('<input type="radio" name="a" id="a1" value="1" disabled>');
            }
            echo('<div id = "t1_text">I can confirm it to be flase.</div>');
            echo('</label>');


            echo('<label class="radio">');
            if ($given_answer_a == 2){
                echo('<input type="radio" name="a" id="a2" value="2" checked disabled>');

            } else {
                echo('<input type="radio" name="a" id="a2" value="2" disabled>');
            }
            echo('<div id = "t2_text">Very likely to be false.</div>');
            echo('</label>');


            echo('<label class="radio">');
            if ($given_answer_a == 3){
                echo('<input type="radio" name="a" id="a3" value="3" checked disabled>');

            } else {
                echo('<input type="radio" name="a" id="a3" value="3" disabled>');
            }
            echo('<div id = "t3_text">Possibly false.</div>');
            echo('</label>');

            echo('<label class="radio">');
            if ($given_answer_a == 4){
                echo('<input type="radio" name="a" id="a4" value="4" checked disabled>');

            } else {
                echo('<input type="radio" name="a" id="a4" value="4" disabled>');
            }
            echo('<div id = "t4_text">Can\'t tell.</div>');
            echo('</label>');


            echo('<label class="radio">');
            if ($given_answer_a == 5){
                echo('<input type="radio" name="a" id="a5" value="5" checked disabled>');

            } else {
                echo('<input type="radio" name="a" id="a5" value="5" disabled>');
            }
            echo('<div id = "t5_text">Possibly true.</div>');
            echo('</label>');

            echo('<label class="radio">');
            if ($given_answer_a == 6){
                echo('<input type="radio" name="a" id="a6" value="6" checked disabled>');

            } else {
                echo('<input type="radio" name="a" id="a6" value="6" disabled>');
            }
            echo('<div id = "t6_text">Very likely to be true.</div>');
            echo('</label>');

            echo('<label class="radio">');
            if ($given_answer_a == 7){
                echo('<input type="radio" name="a" id="a7" value="7" checked disabled>');

            } else {
                echo('<input type="radio" name="a" id="a7" value="7" disabled>');
            }
            echo('<div id = "t7_text">I can confirm it to be true.</div>');
            echo('</label>');


        }

    }
    # end of if condition for "if its a survey question"
            #else{
    if ($page == $npages+1){
        $lastpage=1;
        $tweet_id = 1
        //print("salam");
    ?>
        <p class="lead"><b>Almost there!</b></p>
        <p class="lead">To help us interpret the survey better, please fill in the following questions. 
                        This information is highly valuable for our study. 
                        This information will not be made public and will only be used for academic research purposes.
        </p>

        <!-- <p class="lead"><b><em>News: </em></b><?= linkify_text($tweet_text) ?></p> -->
        <p class="tweet-text lead"></p>
        </div>

        <fieldset>

        <?php
            $is_answered=0;         

            if ($is_answered == 0){  
              echo('<legend>Q1: What is your nationality?</legend>');
              echo('<select  onChange=  "enable_if_done_demographics()" name="demographic_nationality" style="width: 300px; max-width: 300px;">');
              echo('<option style="width: 250px; max-width: 250px;" value="select" >Select</option><option style="width: 250px; max-width: 250px;" value="unitedkingdom" >United Kingdom</option><option style="width: 250px; max-width: 250px;" value="unitedstates" >United States</option><option style="width: 250px; max-width: 250px;" value="prefernotrespond" >Prefer not to respond</option><option style="width: 250px; max-width: 250px;" value="afghanistan" >Afghanistan</option><option style="width: 250px; max-width: 250px;" value="albania" >Albania</option><option style="width: 250px; max-width: 250px;" value="algeria" >Algeria</option><option style="width: 250px; max-width: 250px;" value="angola" >Angola</option><option style="width: 250px; max-width: 250px;" value="antarctica" >Antarctica</option><option style="width: 250px; max-width: 250px;" value="argentina" >Argentina</option><option style="width: 250px; max-width: 250px;" value="armenia" >Armenia</option><option style="width: 250px; max-width: 250px;" value="australia" >Australia</option><option style="width: 250px; max-width: 250px;" value="austria" >Austria</option><option style="width: 250px; max-width: 250px;" value="azerbaijan" >Azerbaijan</option><option style="width: 250px; max-width: 250px;" value="bahamas" >Bahamas</option><option style="width: 250px; max-width: 250px;" value="bangladesh" >Bangladesh</option><option style="width: 250px; max-width: 250px;" value="belarus" >Belarus</option><option style="width: 250px; max-width: 250px;" value="belgium" >Belgium</option><option style="width: 250px; max-width: 250px;" value="belize" >Belize</option><option style="width: 250px; max-width: 250px;" value="benin" >Benin</option><option style="width: 250px; max-width: 250px;" value="bhutan" >Bhutan</option><option style="width: 250px; max-width: 250px;" value="bolivia" >Bolivia</option><option style="width: 250px; max-width: 250px;" value="bosniaandherzegovina" >Bosnia and Herzegovina</option><option style="width: 250px; max-width: 250px;" value="botswana" >Botswana</option><option style="width: 250px; max-width: 250px;" value="brazil" >Brazil</option><option style="width: 250px; max-width: 250px;" value="bruneidarussalam" >Brunei Darussalam</option><option style="width: 250px; max-width: 250px;" value="bulgaria" >Bulgaria</option><option style="width: 250px; max-width: 250px;" value="burkinafaso" >Burkina Faso</option><option style="width: 250px; max-width: 250px;" value="burundi" >Burundi</option><option style="width: 250px; max-width: 250px;" value="cambodia" >Cambodia</option><option style="width: 250px; max-width: 250px;" value="cameroon" >Cameroon</option><option style="width: 250px; max-width: 250px;" value="canada" >Canada</option><option style="width: 250px; max-width: 250px;" value="centralafricanrepublic" >Central African Republic</option><option style="width: 250px; max-width: 250px;" value="chad" >Chad</option><option style="width: 250px; max-width: 250px;" value="chile" >Chile</option><option style="width: 250px; max-width: 250px;" value="china" >China</option><option style="width: 250px; max-width: 250px;" value="colombia" >Colombia</option><option style="width: 250px; max-width: 250px;" value="costarica" >Costa Rica</option><option style="width: 250px; max-width: 250px;" value="cotedivoire" >Cote d\'Ivoire</option><option style="width: 250px; max-width: 250px;" value="croatia" >Croatia</option><option style="width: 250px; max-width: 250px;" value="cuba" >Cuba</option><option style="width: 250px; max-width: 250px;" value="cyprus" >Cyprus</option><option style="width: 250px; max-width: 250px;" value="czechrepublic" >Czech Republic</option><option style="width: 250px; max-width: 250px;" value="demrepkorea" >Dem. Rep. Korea</option><option style="width: 250px; max-width: 250px;" value="democraticrepublicofthecongo" >Democratic Republic of the Congo</option><option style="width: 250px; max-width: 250px;" value="denmark" >Denmark</option><option style="width: 250px; max-width: 250px;" value="djibouti" >Djibouti</option><option style="width: 250px; max-width: 250px;" value="dominicanrepublic" >Dominican Republic</option><option style="width: 250px; max-width: 250px;" value="ecuador" >Ecuador</option><option style="width: 250px; max-width: 250px;" value="egypt" >Egypt</option><option style="width: 250px; max-width: 250px;" value="elsalvador" >El Salvador</option><option style="width: 250px; max-width: 250px;" value="equatorialguinea" >Equatorial Guinea</option><option style="width: 250px; max-width: 250px;" value="eritrea" >Eritrea</option><option style="width: 250px; max-width: 250px;" value="estonia" >Estonia</option><option style="width: 250px; max-width: 250px;" value="ethiopia" >Ethiopia</option><option style="width: 250px; max-width: 250px;" value="falklandislands" >Falkland Islands</option><option style="width: 250px; max-width: 250px;" value="fiji" >Fiji</option><option style="width: 250px; max-width: 250px;" value="finland" >Finland</option><option style="width: 250px; max-width: 250px;" value="france" >France</option><option style="width: 250px; max-width: 250px;" value="frenchsouthernandantarcticlands" >French Southern and Antarctic Lands</option><option style="width: 250px; max-width: 250px;" value="gabon" >Gabon</option><option style="width: 250px; max-width: 250px;" value="georgia" >Georgia</option><option style="width: 250px; max-width: 250px;" value="germany" >Germany</option><option style="width: 250px; max-width: 250px;" value="ghana" >Ghana</option><option style="width: 250px; max-width: 250px;" value="greece" >Greece</option><option style="width: 250px; max-width: 250px;" value="greenland" >Greenland</option><option style="width: 250px; max-width: 250px;" value="guatemala" >Guatemala</option><option style="width: 250px; max-width: 250px;" value="guinea" >Guinea</option><option style="width: 250px; max-width: 250px;" value="guineabissau" >Guinea-Bissau</option><option style="width: 250px; max-width: 250px;" value="guyana" >Guyana</option><option style="width: 250px; max-width: 250px;" value="haiti" >Haiti</option><option style="width: 250px; max-width: 250px;" value="honduras" >Honduras</option><option style="width: 250px; max-width: 250px;" value="hungary" >Hungary</option><option style="width: 250px; max-width: 250px;" value="iceland" >Iceland</option><option style="width: 250px; max-width: 250px;" value="india" >India</option><option style="width: 250px; max-width: 250px;" value="indonesia" >Indonesia</option><option style="width: 250px; max-width: 250px;" value="iran" >Iran</option><option style="width: 250px; max-width: 250px;" value="iraq" >Iraq</option><option style="width: 250px; max-width: 250px;" value="ireland" >Ireland</option><option style="width: 250px; max-width: 250px;" value="israel" >Israel</option><option style="width: 250px; max-width: 250px;" value="italy" >Italy</option><option style="width: 250px; max-width: 250px;" value="jamaica" >Jamaica</option><option style="width: 250px; max-width: 250px;" value="japan" >Japan</option><option style="width: 250px; max-width: 250px;" value="jordan" >Jordan</option><option style="width: 250px; max-width: 250px;" value="kazakhstan" >Kazakhstan</option><option style="width: 250px; max-width: 250px;" value="kenya" >Kenya</option><option style="width: 250px; max-width: 250px;" value="kosovo" >Kosovo</option><option style="width: 250px; max-width: 250px;" value="kuwait" >Kuwait</option><option style="width: 250px; max-width: 250px;" value="kyrgyzstan" >Kyrgyzstan</option><option style="width: 250px; max-width: 250px;" value="laopdr" >Lao PDR</option><option style="width: 250px; max-width: 250px;" value="latvia" >Latvia</option><option style="width: 250px; max-width: 250px;" value="lebanon" >Lebanon</option><option style="width: 250px; max-width: 250px;" value="lesotho" >Lesotho</option><option style="width: 250px; max-width: 250px;" value="liberia" >Liberia</option><option style="width: 250px; max-width: 250px;" value="libya" >Libya</option><option style="width: 250px; max-width: 250px;" value="lithuania" >Lithuania</option><option style="width: 250px; max-width: 250px;" value="luxembourg" >Luxembourg</option><option style="width: 250px; max-width: 250px;" value="macedonia" >Macedonia</option><option style="width: 250px; max-width: 250px;" value="madagascar" >Madagascar</option><option style="width: 250px; max-width: 250px;" value="malawi" >Malawi</option><option style="width: 250px; max-width: 250px;" value="malaysia" >Malaysia</option><option style="width: 250px; max-width: 250px;" value="mali" >Mali</option><option style="width: 250px; max-width: 250px;" value="mauritania" >Mauritania</option><option style="width: 250px; max-width: 250px;" value="mexico" >Mexico</option><option style="width: 250px; max-width: 250px;" value="moldova" >Moldova</option><option style="width: 250px; max-width: 250px;" value="mongolia" >Mongolia</option><option style="width: 250px; max-width: 250px;" value="montenegro" >Montenegro</option><option style="width: 250px; max-width: 250px;" value="morocco" >Morocco</option><option style="width: 250px; max-width: 250px;" value="mozambique" >Mozambique</option><option style="width: 250px; max-width: 250px;" value="myanmar" >Myanmar</option><option style="width: 250px; max-width: 250px;" value="namibia" >Namibia</option><option style="width: 250px; max-width: 250px;" value="nepal" >Nepal</option><option style="width: 250px; max-width: 250px;" value="netherlands" >Netherlands</option><option style="width: 250px; max-width: 250px;" value="newcaledonia" >New Caledonia</option><option style="width: 250px; max-width: 250px;" value="newzealand" >New Zealand</option><option style="width: 250px; max-width: 250px;" value="nicaragua" >Nicaragua</option><option style="width: 250px; max-width: 250px;" value="niger" >Niger</option><option style="width: 250px; max-width: 250px;" value="nigeria" >Nigeria</option><option style="width: 250px; max-width: 250px;" value="northerncyprus" >Northern Cyprus</option><option style="width: 250px; max-width: 250px;" value="norway" >Norway</option><option style="width: 250px; max-width: 250px;" value="oman" >Oman</option><option style="width: 250px; max-width: 250px;" value="pakistan" >Pakistan</option><option style="width: 250px; max-width: 250px;" value="palestine" >Palestine</option><option style="width: 250px; max-width: 250px;" value="panama" >Panama</option><option style="width: 250px; max-width: 250px;" value="papuanewguinea" >Papua New Guinea</option><option style="width: 250px; max-width: 250px;" value="paraguay" >Paraguay</option><option style="width: 250px; max-width: 250px;" value="peru" >Peru</option><option style="width: 250px; max-width: 250px;" value="philippines" >Philippines</option><option style="width: 250px; max-width: 250px;" value="poland" >Poland</option><option style="width: 250px; max-width: 250px;" value="portugal" >Portugal</option><option style="width: 250px; max-width: 250px;" value="puertorico" >Puerto Rico</option><option style="width: 250px; max-width: 250px;" value="qatar" >Qatar</option><option style="width: 250px; max-width: 250px;" value="republicofcongo" >Republic of Congo</option><option style="width: 250px; max-width: 250px;" value="republicofkorea" >Republic of Korea</option><option style="width: 250px; max-width: 250px;" value="romania" >Romania</option><option style="width: 250px; max-width: 250px;" value="russianfederation" >Russian Federation</option><option style="width: 250px; max-width: 250px;" value="rwanda" >Rwanda</option><option style="width: 250px; max-width: 250px;" value="saudiarabia" >Saudi Arabia</option><option style="width: 250px; max-width: 250px;" value="senegal" >Senegal</option><option style="width: 250px; max-width: 250px;" value="serbia" >Serbia</option><option style="width: 250px; max-width: 250px;" value="sierraleone" >Sierra Leone</option><option style="width: 250px; max-width: 250px;" value="slovakia" >Slovakia</option><option style="width: 250px; max-width: 250px;" value="slovenia" >Slovenia</option><option style="width: 250px; max-width: 250px;" value="solomonislands" >Solomon Islands</option><option style="width: 250px; max-width: 250px;" value="somalia" >Somalia</option><option style="width: 250px; max-width: 250px;" value="somaliland" >Somaliland</option><option style="width: 250px; max-width: 250px;" value="southafrica" >South Africa</option><option style="width: 250px; max-width: 250px;" value="southsudan" >South Sudan</option><option style="width: 250px; max-width: 250px;" value="spain" >Spain</option><option style="width: 250px; max-width: 250px;" value="srilanka" >Sri Lanka</option><option style="width: 250px; max-width: 250px;" value="sudan" >Sudan</option><option style="width: 250px; max-width: 250px;" value="suriname" >Suriname</option><option style="width: 250px; max-width: 250px;" value="swaziland" >Swaziland</option><option style="width: 250px; max-width: 250px;" value="sweden" >Sweden</option><option style="width: 250px; max-width: 250px;" value="switzerland" >Switzerland</option><option style="width: 250px; max-width: 250px;" value="syria" >Syria</option><option style="width: 250px; max-width: 250px;" value="taiwan" >Taiwan</option><option style="width: 250px; max-width: 250px;" value="tajikistan" >Tajikistan</option><option style="width: 250px; max-width: 250px;" value="tanzania" >Tanzania</option><option style="width: 250px; max-width: 250px;" value="thailand" >Thailand</option><option style="width: 250px; max-width: 250px;" value="thegambia" >The Gambia</option><option style="width: 250px; max-width: 250px;" value="timorleste" >Timor-Leste</option><option style="width: 250px; max-width: 250px;" value="togo" >Togo</option><option style="width: 250px; max-width: 250px;" value="trinidadandtobago" >Trinidad and Tobago</option><option style="width: 250px; max-width: 250px;" value="tunisia" >Tunisia</option><option style="width: 250px; max-width: 250px;" value="turkey" >Turkey</option><option style="width: 250px; max-width: 250px;" value="turkmenistan" >Turkmenistan</option><option style="width: 250px; max-width: 250px;" value="uganda" >Uganda</option><option style="width: 250px; max-width: 250px;" value="ukraine" >Ukraine</option><option style="width: 250px; max-width: 250px;" value="unitedarabemirates" >United Arab Emirates</option><option style="width: 250px; max-width: 250px;" value="uruguay" >Uruguay</option><option style="width: 250px; max-width: 250px;" value="uzbekistan" >Uzbekistan</option><option style="width: 250px; max-width: 250px;" value="vanuatu" >Vanuatu</option><option style="width: 250px; max-width: 250px;" value="venezuela" >Venezuela</option><option style="width: 250px; max-width: 250px;" value="vietnam" >Vietnam</option><option style="width: 250px; max-width: 250px;" value="westernsahara" >Western Sahara</option><option style="width: 250px; max-width: 250px;" value="yemen" >Yemen</option><option style="width: 250px; max-width: 250px;" value="zambia" >Zambia</option><option style="width: 250px; max-width: 250px;" value="zimbabwe" >Zimbabwe</option></select>');

              echo('<br><br><legend>Q2: What is your country of residence?</legend>');
              echo('<select  onChange=  "enable_if_done_demographics()"  name="demographic_residence" style="width: 300px; max-width: 300px;">');
              echo('<option style="width: 250px; max-width: 250px;" value="select" >Select</option><option style="width: 250px; max-width: 250px;" value="unitedkingdom" >United Kingdom</option><option style="width: 250px; max-width: 250px;" value="unitedstates" >United States</option><option style="width: 250px; max-width: 250px;" value="prefernotrespond" >Prefer not to respond</option><option style="width: 250px; max-width: 250px;" value="afghanistan" >Afghanistan</option><option style="width: 250px; max-width: 250px;" value="albania" >Albania</option><option style="width: 250px; max-width: 250px;" value="algeria" >Algeria</option><option style="width: 250px; max-width: 250px;" value="angola" >Angola</option><option style="width: 250px; max-width: 250px;" value="antarctica" >Antarctica</option><option style="width: 250px; max-width: 250px;" value="argentina" >Argentina</option><option style="width: 250px; max-width: 250px;" value="armenia" >Armenia</option><option style="width: 250px; max-width: 250px;" value="australia" >Australia</option><option style="width: 250px; max-width: 250px;" value="austria" >Austria</option><option style="width: 250px; max-width: 250px;" value="azerbaijan" >Azerbaijan</option><option style="width: 250px; max-width: 250px;" value="bahamas" >Bahamas</option><option style="width: 250px; max-width: 250px;" value="bangladesh" >Bangladesh</option><option style="width: 250px; max-width: 250px;" value="belarus" >Belarus</option><option style="width: 250px; max-width: 250px;" value="belgium" >Belgium</option><option style="width: 250px; max-width: 250px;" value="belize" >Belize</option><option style="width: 250px; max-width: 250px;" value="benin" >Benin</option><option style="width: 250px; max-width: 250px;" value="bhutan" >Bhutan</option><option style="width: 250px; max-width: 250px;" value="bolivia" >Bolivia</option><option style="width: 250px; max-width: 250px;" value="bosniaandherzegovina" >Bosnia and Herzegovina</option><option style="width: 250px; max-width: 250px;" value="botswana" >Botswana</option><option style="width: 250px; max-width: 250px;" value="brazil" >Brazil</option><option style="width: 250px; max-width: 250px;" value="bruneidarussalam" >Brunei Darussalam</option><option style="width: 250px; max-width: 250px;" value="bulgaria" >Bulgaria</option><option style="width: 250px; max-width: 250px;" value="burkinafaso" >Burkina Faso</option><option style="width: 250px; max-width: 250px;" value="burundi" >Burundi</option><option style="width: 250px; max-width: 250px;" value="cambodia" >Cambodia</option><option style="width: 250px; max-width: 250px;" value="cameroon" >Cameroon</option><option style="width: 250px; max-width: 250px;" value="canada" >Canada</option><option style="width: 250px; max-width: 250px;" value="centralafricanrepublic" >Central African Republic</option><option style="width: 250px; max-width: 250px;" value="chad" >Chad</option><option style="width: 250px; max-width: 250px;" value="chile" >Chile</option><option style="width: 250px; max-width: 250px;" value="china" >China</option><option style="width: 250px; max-width: 250px;" value="colombia" >Colombia</option><option style="width: 250px; max-width: 250px;" value="costarica" >Costa Rica</option><option style="width: 250px; max-width: 250px;" value="cotedivoire" >Cote d\'Ivoire</option><option style="width: 250px; max-width: 250px;" value="croatia" >Croatia</option><option style="width: 250px; max-width: 250px;" value="cuba" >Cuba</option><option style="width: 250px; max-width: 250px;" value="cyprus" >Cyprus</option><option style="width: 250px; max-width: 250px;" value="czechrepublic" >Czech Republic</option><option style="width: 250px; max-width: 250px;" value="demrepkorea" >Dem. Rep. Korea</option><option style="width: 250px; max-width: 250px;" value="democraticrepublicofthecongo" >Democratic Republic of the Congo</option><option style="width: 250px; max-width: 250px;" value="denmark" >Denmark</option><option style="width: 250px; max-width: 250px;" value="djibouti" >Djibouti</option><option style="width: 250px; max-width: 250px;" value="dominicanrepublic" >Dominican Republic</option><option style="width: 250px; max-width: 250px;" value="ecuador" >Ecuador</option><option style="width: 250px; max-width: 250px;" value="egypt" >Egypt</option><option style="width: 250px; max-width: 250px;" value="elsalvador" >El Salvador</option><option style="width: 250px; max-width: 250px;" value="equatorialguinea" >Equatorial Guinea</option><option style="width: 250px; max-width: 250px;" value="eritrea" >Eritrea</option><option style="width: 250px; max-width: 250px;" value="estonia" >Estonia</option><option style="width: 250px; max-width: 250px;" value="ethiopia" >Ethiopia</option><option style="width: 250px; max-width: 250px;" value="falklandislands" >Falkland Islands</option><option style="width: 250px; max-width: 250px;" value="fiji" >Fiji</option><option style="width: 250px; max-width: 250px;" value="finland" >Finland</option><option style="width: 250px; max-width: 250px;" value="france" >France</option><option style="width: 250px; max-width: 250px;" value="frenchsouthernandantarcticlands" >French Southern and Antarctic Lands</option><option style="width: 250px; max-width: 250px;" value="gabon" >Gabon</option><option style="width: 250px; max-width: 250px;" value="georgia" >Georgia</option><option style="width: 250px; max-width: 250px;" value="germany" >Germany</option><option style="width: 250px; max-width: 250px;" value="ghana" >Ghana</option><option style="width: 250px; max-width: 250px;" value="greece" >Greece</option><option style="width: 250px; max-width: 250px;" value="greenland" >Greenland</option><option style="width: 250px; max-width: 250px;" value="guatemala" >Guatemala</option><option style="width: 250px; max-width: 250px;" value="guinea" >Guinea</option><option style="width: 250px; max-width: 250px;" value="guineabissau" >Guinea-Bissau</option><option style="width: 250px; max-width: 250px;" value="guyana" >Guyana</option><option style="width: 250px; max-width: 250px;" value="haiti" >Haiti</option><option style="width: 250px; max-width: 250px;" value="honduras" >Honduras</option><option style="width: 250px; max-width: 250px;" value="hungary" >Hungary</option><option style="width: 250px; max-width: 250px;" value="iceland" >Iceland</option><option style="width: 250px; max-width: 250px;" value="india" >India</option><option style="width: 250px; max-width: 250px;" value="indonesia" >Indonesia</option><option style="width: 250px; max-width: 250px;" value="iran" >Iran</option><option style="width: 250px; max-width: 250px;" value="iraq" >Iraq</option><option style="width: 250px; max-width: 250px;" value="ireland" >Ireland</option><option style="width: 250px; max-width: 250px;" value="israel" >Israel</option><option style="width: 250px; max-width: 250px;" value="italy" >Italy</option><option style="width: 250px; max-width: 250px;" value="jamaica" >Jamaica</option><option style="width: 250px; max-width: 250px;" value="japan" >Japan</option><option style="width: 250px; max-width: 250px;" value="jordan" >Jordan</option><option style="width: 250px; max-width: 250px;" value="kazakhstan" >Kazakhstan</option><option style="width: 250px; max-width: 250px;" value="kenya" >Kenya</option><option style="width: 250px; max-width: 250px;" value="kosovo" >Kosovo</option><option style="width: 250px; max-width: 250px;" value="kuwait" >Kuwait</option><option style="width: 250px; max-width: 250px;" value="kyrgyzstan" >Kyrgyzstan</option><option style="width: 250px; max-width: 250px;" value="laopdr" >Lao PDR</option><option style="width: 250px; max-width: 250px;" value="latvia" >Latvia</option><option style="width: 250px; max-width: 250px;" value="lebanon" >Lebanon</option><option style="width: 250px; max-width: 250px;" value="lesotho" >Lesotho</option><option style="width: 250px; max-width: 250px;" value="liberia" >Liberia</option><option style="width: 250px; max-width: 250px;" value="libya" >Libya</option><option style="width: 250px; max-width: 250px;" value="lithuania" >Lithuania</option><option style="width: 250px; max-width: 250px;" value="luxembourg" >Luxembourg</option><option style="width: 250px; max-width: 250px;" value="macedonia" >Macedonia</option><option style="width: 250px; max-width: 250px;" value="madagascar" >Madagascar</option><option style="width: 250px; max-width: 250px;" value="malawi" >Malawi</option><option style="width: 250px; max-width: 250px;" value="malaysia" >Malaysia</option><option style="width: 250px; max-width: 250px;" value="mali" >Mali</option><option style="width: 250px; max-width: 250px;" value="mauritania" >Mauritania</option><option style="width: 250px; max-width: 250px;" value="mexico" >Mexico</option><option style="width: 250px; max-width: 250px;" value="moldova" >Moldova</option><option style="width: 250px; max-width: 250px;" value="mongolia" >Mongolia</option><option style="width: 250px; max-width: 250px;" value="montenegro" >Montenegro</option><option style="width: 250px; max-width: 250px;" value="morocco" >Morocco</option><option style="width: 250px; max-width: 250px;" value="mozambique" >Mozambique</option><option style="width: 250px; max-width: 250px;" value="myanmar" >Myanmar</option><option style="width: 250px; max-width: 250px;" value="namibia" >Namibia</option><option style="width: 250px; max-width: 250px;" value="nepal" >Nepal</option><option style="width: 250px; max-width: 250px;" value="netherlands" >Netherlands</option><option style="width: 250px; max-width: 250px;" value="newcaledonia" >New Caledonia</option><option style="width: 250px; max-width: 250px;" value="newzealand" >New Zealand</option><option style="width: 250px; max-width: 250px;" value="nicaragua" >Nicaragua</option><option style="width: 250px; max-width: 250px;" value="niger" >Niger</option><option style="width: 250px; max-width: 250px;" value="nigeria" >Nigeria</option><option style="width: 250px; max-width: 250px;" value="northerncyprus" >Northern Cyprus</option><option style="width: 250px; max-width: 250px;" value="norway" >Norway</option><option style="width: 250px; max-width: 250px;" value="oman" >Oman</option><option style="width: 250px; max-width: 250px;" value="pakistan" >Pakistan</option><option style="width: 250px; max-width: 250px;" value="palestine" >Palestine</option><option style="width: 250px; max-width: 250px;" value="panama" >Panama</option><option style="width: 250px; max-width: 250px;" value="papuanewguinea" >Papua New Guinea</option><option style="width: 250px; max-width: 250px;" value="paraguay" >Paraguay</option><option style="width: 250px; max-width: 250px;" value="peru" >Peru</option><option style="width: 250px; max-width: 250px;" value="philippines" >Philippines</option><option style="width: 250px; max-width: 250px;" value="poland" >Poland</option><option style="width: 250px; max-width: 250px;" value="portugal" >Portugal</option><option style="width: 250px; max-width: 250px;" value="puertorico" >Puerto Rico</option><option style="width: 250px; max-width: 250px;" value="qatar" >Qatar</option><option style="width: 250px; max-width: 250px;" value="republicofcongo" >Republic of Congo</option><option style="width: 250px; max-width: 250px;" value="republicofkorea" >Republic of Korea</option><option style="width: 250px; max-width: 250px;" value="romania" >Romania</option><option style="width: 250px; max-width: 250px;" value="russianfederation" >Russian Federation</option><option style="width: 250px; max-width: 250px;" value="rwanda" >Rwanda</option><option style="width: 250px; max-width: 250px;" value="saudiarabia" >Saudi Arabia</option><option style="width: 250px; max-width: 250px;" value="senegal" >Senegal</option><option style="width: 250px; max-width: 250px;" value="serbia" >Serbia</option><option style="width: 250px; max-width: 250px;" value="sierraleone" >Sierra Leone</option><option style="width: 250px; max-width: 250px;" value="slovakia" >Slovakia</option><option style="width: 250px; max-width: 250px;" value="slovenia" >Slovenia</option><option style="width: 250px; max-width: 250px;" value="solomonislands" >Solomon Islands</option><option style="width: 250px; max-width: 250px;" value="somalia" >Somalia</option><option style="width: 250px; max-width: 250px;" value="somaliland" >Somaliland</option><option style="width: 250px; max-width: 250px;" value="southafrica" >South Africa</option><option style="width: 250px; max-width: 250px;" value="southsudan" >South Sudan</option><option style="width: 250px; max-width: 250px;" value="spain" >Spain</option><option style="width: 250px; max-width: 250px;" value="srilanka" >Sri Lanka</option><option style="width: 250px; max-width: 250px;" value="sudan" >Sudan</option><option style="width: 250px; max-width: 250px;" value="suriname" >Suriname</option><option style="width: 250px; max-width: 250px;" value="swaziland" >Swaziland</option><option style="width: 250px; max-width: 250px;" value="sweden" >Sweden</option><option style="width: 250px; max-width: 250px;" value="switzerland" >Switzerland</option><option style="width: 250px; max-width: 250px;" value="syria" >Syria</option><option style="width: 250px; max-width: 250px;" value="taiwan" >Taiwan</option><option style="width: 250px; max-width: 250px;" value="tajikistan" >Tajikistan</option><option style="width: 250px; max-width: 250px;" value="tanzania" >Tanzania</option><option style="width: 250px; max-width: 250px;" value="thailand" >Thailand</option><option style="width: 250px; max-width: 250px;" value="thegambia" >The Gambia</option><option style="width: 250px; max-width: 250px;" value="timorleste" >Timor-Leste</option><option style="width: 250px; max-width: 250px;" value="togo" >Togo</option><option style="width: 250px; max-width: 250px;" value="trinidadandtobago" >Trinidad and Tobago</option><option style="width: 250px; max-width: 250px;" value="tunisia" >Tunisia</option><option style="width: 250px; max-width: 250px;" value="turkey" >Turkey</option><option style="width: 250px; max-width: 250px;" value="turkmenistan" >Turkmenistan</option><option style="width: 250px; max-width: 250px;" value="uganda" >Uganda</option><option style="width: 250px; max-width: 250px;" value="ukraine" >Ukraine</option><option style="width: 250px; max-width: 250px;" value="unitedarabemirates" >United Arab Emirates</option><option style="width: 250px; max-width: 250px;" value="uruguay" >Uruguay</option><option style="width: 250px; max-width: 250px;" value="uzbekistan" >Uzbekistan</option><option style="width: 250px; max-width: 250px;" value="vanuatu" >Vanuatu</option><option style="width: 250px; max-width: 250px;" value="venezuela" >Venezuela</option><option style="width: 250px; max-width: 250px;" value="vietnam" >Vietnam</option><option style="width: 250px; max-width: 250px;" value="westernsahara" >Western Sahara</option><option style="width: 250px; max-width: 250px;" value="yemen" >Yemen</option><option style="width: 250px; max-width: 250px;" value="zambia" >Zambia</option><option style="width: 250px; max-width: 250px;" value="zimbabwe" >Zimbabwe</option></select>');

              echo('<br><br><legend>Q3: Which of the following best identifies yours gender?</legend>');
              echo('<select  onChange=  "enable_if_done_demographics()"  name="demographic_gender" style="width: 300px; max-width: 300px;">');
              echo('<option style="width: 250px; max-width: 250px;" value="select" >Select</option><option style="width: 250px; max-width: 250px;" value="female" >Female</option><option style="width: 250px; max-width: 250px;" value="male" >Male</option><option style="width: 250px; max-width: 250px;" value="other" >Other</option><option style="width: 250px; max-width: 250px;" value="prefernotrespond" >Prefer not to respond</option></select>');

              echo('<br><br><legend>Q4: What is your age?</legend>');
              echo('<select  onChange=  "enable_if_done_demographics()"  name="demographic_age" style="width: 300px; max-width: 300px;">');
              echo('<option style="width: 250px; max-width: 250px;" value="select" >Select</option><option style="width: 250px; max-width: 250px;" value="under12" >Under 12 years old</option><option style="width: 250px; max-width: 250px;" value="12-17" >12-17 years old</option><option style="width: 250px; max-width: 250px;" value="18-24" >18-24 years old</option><option style="width: 250px; max-width: 250px;" value="25-34" >25-34 years old</option><option style="width: 250px; max-width: 250px;" value="35-44" >35-44 years old</option><option style="width: 250px; max-width: 250px;" value="45-54" >45-54 years old</option><option style="width: 250px; max-width: 250px;" value="55-64" >55-64 years old</option><option style="width: 250px; max-width: 250px;" value="65-74" >65-74 years old</option><option style="width: 250px; max-width: 250px;" value="75orabove" >75 years or older</option><option style="width: 250px; max-width: 250px;" value="prefernotrespond" >Prefer not to respond</option></select>');

              echo('<br><br><legend>Q5: What is the highest degree or level of school you have completed? If currently enrolled, highest degree received.</legend>');
              echo('<select  onChange=  "enable_if_done_demographics()"  name="demographic_degree" style="width: 300px; max-width: 300px;">');
              echo('<option style="width: 250px; max-width: 250px;" value="select" >Select</option><option style="width: 250px; max-width: 250px;" value="noneorgrade18" >None, or grade 1-8</option><option style="width: 250px; max-width: 250px;" value="highschoolincompletegrades911" >High school incomplete (Grades 9-11)</option><option style="width: 250px; max-width: 250px;" value="highschoolgraduategrade12orgedcertificate" >High school graduate (Grade 12 or GED certificate)</option><option style="width: 250px; max-width: 250px;" value="technicaltradeorvocationalschoolafterhighschool" >Technical, trade, or vocational school AFTER high school</option><option style="width: 250px; max-width: 250px;" value="somecollegeassociatedegreeno4yeardegree" >Some college, associate degree, no 4-year degree</option><option style="width: 250px; max-width: 250px;" value="collegegraduatebsbaorother4yeardegree" >College graduate (B.S., B.A., or other 4-year degree)</option><option style="width: 250px; max-width: 250px;" value="postgraduatetrainingorprofessionalschoolingaftercollegeegtowardamastersdegreeorphdlawormedicalschool" > Post-graduate training or professional schooling after college (e.g., toward a master\'s Degree or Ph.D.; law or medical school)</option><option style="width: 250px; max-width: 250px;" value="prefernotrespond" >Prefer not to respond</option></select>');

              echo('<br><br><legend>Q6: What is your employment status?</legend>');
              echo('<select  onChange=  "enable_if_done_demographics()"  name="demographic_employment" style="width: 300px; max-width: 300px;">');
              echo('<option style="width: 250px; max-width: 250px;" value="select" >Select</option><option style="width: 250px; max-width: 250px;" value="infulltimeworkpermanent" >In full time work, permanent</option><option style="width: 250px; max-width: 250px;" value="infulltimeworktempcontract" >In full time work, temp/contract</option><option style="width: 250px; max-width: 250px;" value="inparttimeworkpermanent" >In part time work, permanent</option><option style="width: 250px; max-width: 250px;" value="inparttimeworktempcontract" >In part time work, temp/contract</option><option style="width: 250px; max-width: 250px;" value="parttimeworkparttimestudent" >Part time work, part time student</option><option style="width: 250px; max-width: 250px;" value="studentonly" >Student only</option><option style="width: 250px; max-width: 250px;" value="unemployed" >Unemployed</option><option style="width: 250px; max-width: 250px;" value="incapacity" >Incapacity</option><option style="width: 250px; max-width: 250px;" value="retired" >Retired</option><option style="width: 250px; max-width: 250px;" value="selfemployed" >Self-employed</option><option style="width: 250px; max-width: 250px;" value="prefernotrespond" >Prefer not to respond</option></select>');

              echo('<br><br><legend>Q7: What is your income (approximate US $ per annum)?</legend>');
              echo('<select  onChange=  "enable_if_done_demographics()"  name="demographic_income" style="width: 300px; max-width: 300px;">');
              echo('<option style="width: 250px; max-width: 250px;" value="select" >Select</option><option style="width: 250px; max-width: 250px;" value="under10000" >Under 10,000</option><option style="width: 250px; max-width: 250px;" value="10000ndash20000" >10,000&ndash;20,000</option><option style="width: 250px; max-width: 250px;" value="20001ndash30000" >20,001&ndash;30,000</option><option style="width: 250px; max-width: 250px;" value="30001ndash40000" >30,001&ndash;40,000</option><option style="width: 250px; max-width: 250px;" value="40001ndash50000" >40,001&ndash;50,000</option><option style="width: 250px; max-width: 250px;" value="50001ndash60000" >50,001&ndash;60,000</option><option style="width: 250px; max-width: 250px;" value="60001ndash70000" >60,001&ndash;70,000</option><option style="width: 250px; max-width: 250px;" value="70001ndash100000" >70,001&ndash;100,000</option><option style="width: 250px; max-width: 250px;" value="100001ndash150000" >100,001&ndash;150,000</option><option style="width: 250px; max-width: 250px;" value="150001ormore" >150,001 or more</option><option style="width: 250px; max-width: 250px;" value="prefernotrespond" >Prefer not to respond</option></select>');

              echo('<br><br><legend>Q8: Which of the following best defines your political view?</legend>');
              echo('<select  onChange=  "enable_if_done_demographics()"  name="demographic_political_view" style="width: 300px; max-width: 300px;">');
              echo('<option style="width: 250px; max-width: 250px;" value="select" >Select</option><option style="width: 250px; max-width: 250px;" value="veryconservative" >Very conservative</option><option style="width: 250px; max-width: 250px;" value="conservative" >Conservative</option><option style="width: 250px; max-width: 250px;" value="moderate" >Moderate</option><option style="width: 250px; max-width: 250px;" value="liberal" >Liberal</option><option style="width: 250px; max-width: 250px;" value="veryliberal" >Very liberal</option><option style="width: 250px; max-width: 250px;" value="other" >Other</option><option style="width: 250px; max-width: 250px;" value="dontknow" >Don\'t know</option><option style="width: 250px; max-width: 250px;" value="prefernotrespond" >Prefer not to respond</option></select>');

              echo('<br><br><legend>Q9: Please specify your race/ethnicity:</legend>');
              echo('<select  onChange=  "enable_if_done_demographics()"  name="demographic_race" style="width: 300px; max-width: 300px;">');
              echo('<option style="width: 250px; max-width: 250px;" value="select" >Select</option><option style="width: 250px; max-width: 250px;" value="americanindianoralaskanative" >American Indian or Alaska Native</option><option style="width: 250px; max-width: 250px;" value="asian" >Asian</option><option style="width: 250px; max-width: 250px;" value="blackorafricanamerican" >Black or African American</option><option style="width: 250px; max-width: 250px;" value="hispanicorlatino" >Hispanic or Latino</option><option style="width: 250px; max-width: 250px;" value="nativehawaiianorotherpacificislander" >Native Hawaiian or Other Pacific Islander</option><option style="width: 250px; max-width: 250px;" value="white" >White</option><option style="width: 250px; max-width: 250px;" value="other" >Other</option><option style="width: 250px; max-width: 250px;" value="dontknow" >Don\'t know</option><option style="width: 250px; max-width: 250px;" value="prefernotrespond" >Prefer not to respond</option></select>');

              echo('<br><br><legend>Q10: What is your marital status?</legend>');
              echo('<select  onChange=  "enable_if_done_demographics()"  name="demographic_marital_status" style="width: 300px; max-width: 300px;">');
              echo('<option style="width: 250px; max-width: 250px;" value="select" >Select</option><option style="width: 250px; max-width: 250px;" value="single" >Single (never been married)</option><option style="width: 250px; max-width: 250px;" value="living_with_partner" >Living with a Partner</option><option style="width: 250px; max-width: 250px;" value="married" >Married</option><option style="width: 250px; max-width: 250px;" value="separated" >Separated</option><option style="width: 250px; max-width: 250px;" value="widowed" >Widowed</option><option style="width: 250px; max-width: 250px;" value="divorced" >Divorced</option><option style="width: 250px; max-width: 250px;" value="prefernotrespond" >Prefer not to respond</option></select>');
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
$set_number = 1;
$set_number = "ap";
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
        $demographic_nationality =  get_default($_GET, "demographic_nationality", " ");
        $demographic_residence =  get_default($_GET, "demographic_residence", " ");
        $demographic_gender =  get_default($_GET, "demographic_gender", " ");
        $demographic_age =  get_default($_GET, "demographic_age", " ");
        $demographic_degree =  get_default($_GET, "demographic_degree", " ");
        $demographic_employment =  get_default($_GET, "demographic_employment", " ");
        $demographic_income =  get_default($_GET, "demographic_income", " ");
        $demographic_political_view  =  get_default($_GET, "demographic_political_view", " ");
        $demographic_race =  get_default($_GET, "demographic_race", " ");
        $demographic_marital_status =  get_default($_GET, "demographic_marital_status", " ");

        // $prior_knowledge_heard =  get_default($_GET, "prior_knowledge_heard", " ");
        // $prior_knowledge_your_job =  get_default($_GET, "prior_knowledge_your_job", " ");
        // $prior_knowledge_close_job =  get_default($_GET, "prior_knowledge_close_job", " ");
        // $prior_knowledge_heard_description =  get_default($_GET, "prior_knowledge_heard_description", " ");




        survey_demographics($workerid,$demographic_nationality, $demographic_residence, $demographic_gender, $demographic_age, $demographic_degree, $demographic_employment, $demographic_income, $demographic_political_view , $demographic_race, $demographic_marital_status);


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

    if (document.getElementById("a3").checked){
        ra = 3;
    }

    if (document.getElementById("a4").checked){
        ra = 4;
    }

        if (document.getElementById("a5").checked){
        ra = 5;
    }

        if (document.getElementById("a6").checked){
        ra = 6;
    }

        if (document.getElementById("a7").checked){
        ra = 7;
    }

    // if (document.getElementById("a5").checked){
    //     ra = 5;
    // }



    // if (document.getElementById("b1").checked){
    //      rb = 1;
    // }

    // if (document.getElementById("b2").checked){
    //      rb = 2;
    // }


    // if (document.getElementById("b3").checked){
    //      rb = 3;
    // }

    // if (document.getElementById("b4").checked){
    //      rb = 4;
    // }

    //if (document.getElementById("b4").checked){
    //     rb = 4;
    //}


    //if (document.getElementById("b5").checked){
    //     rb = 5;
    //}



    //if (document.getElementById("c1").checked){
    //    rc = 1;
    //}

    //if (document.getElementById("c2").checked){
    //    rc = 2;
    //}

    //if (document.getElementById("text_1").checked){
    //text = document.getElementById("text_1").value;
    text = "";
    //}


    callNext(p, w, ra, rb, rc, rd,text);



}


// function sendQueryWithAnswerLast(p, w)
// {

//     ra = 0;
//     rb = 0;
//     rc = 0;
//     rd = 0;


//     if (document.getElementById("a1").checked){
//         ra = 1;
//     }

//     if (document.getElementById("a2").checked){
//         ra = 2;
//     }

//     if (document.getElementById("a3").checked){
//         ra = 3;
//     }

//     if (document.getElementById("a4").checked){
//         ra = 4;
//     }



//     callNext(p, w, ra, rb, rc, rd);



// }


function sendQueryWithAnswerLast(p, w)
{

    // demographic_nationality =  " ";
    // demographic_residence =  " ";
    // demographic_gender = " ";
    // demographic_age =  " ";
    // demographic_degree = " ";
    // demographic_employment =   " ";
    // demographic_income =   " ";
    // demographic_political_view  =   " ";
    // demographic_race =   " ";
    // demographic_marital_status =   " ";

    demographic_nationality = document.getElementsByName('demographic_nationality')[0].value;
    demographic_residence = document.getElementsByName('demographic_residence')[0].value;
    demographic_gender = document.getElementsByName('demographic_gender')[0].value;
    demographic_age = document.getElementsByName('demographic_age')[0].value;
    demographic_degree = document.getElementsByName('demographic_degree')[0].value;
    demographic_employment = document.getElementsByName('demographic_employment')[0].value;
    demographic_income = document.getElementsByName('demographic_income')[0].value;
    demographic_political_view = document.getElementsByName('demographic_political_view')[0].value;
    demographic_race = document.getElementsByName('demographic_race')[0].value;
    demographic_marital_status = document.getElementsByName('demographic_marital_status')[0].value;

    // prior_knowledge_heard = document.getElementsByName('prior_knowledge_heard')[0].value;
    // prior_knowledge_your_job = document.getElementsByName('prior_knowledge_your_job')[0].value;
    // prior_knowledge_close_job = document.getElementsByName('prior_knowledge_close_job')[0].value;
    // prior_knowledge_heard_description = document.getElementsByName('prior_knowledge_heard_description')[0].value;
    
    
    // callNext(p, w);
    // , demographic_nationality, demographic_residence, demographic_gender, demographic_age,  demographic_degree, demographic_employment, demographic_income, demographic_political_view , demographic_race, demographic_marital_status);
    callLast(p, w, demographic_nationality, demographic_residence, demographic_gender, demographic_age,  demographic_degree, demographic_employment, demographic_income, demographic_political_view , demographic_race, demographic_marital_status);
}

function selection_finished_demographics(){


if (document.getElementsByName('demographic_nationality')[0].value == 'select') {
    return false;    
}
if (document.getElementsByName('demographic_residence')[0].value == 'select') {
    return false;    
}
if (document.getElementsByName('demographic_gender')[0].value == 'select') {
    return false;   
}
if (document.getElementsByName('demographic_age')[0].value == 'select') {
    return false;   
}
if (document.getElementsByName('demographic_degree')[0].value == 'select') {
    return false; 
}
if (document.getElementsByName('demographic_employment')[0].value == 'select') {
    return false;    
}
if (document.getElementsByName('demographic_income')[0].value == 'select') {
    return false;  
}
if (document.getElementsByName('demographic_political_view')[0].value == 'select') {
    return false;    
}
if (document.getElementsByName('demographic_race')[0].value == 'select') {
    return false;
}
if (document.getElementsByName('demographic_marital_status')[0].value == 'select') {
    return false;   
}
return true;
}

function enable_if_done_demographics() {
if (selection_finished_demographics()){
document.getElementById("btn_submit").disabled = false;
}
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
