 <?php

// Get value from array or default if not available
function get_default($arr, $key, $default)
{
    if (array_key_exists($key, $arr)) {
        return $arr[$key];
    }

    return $default;
}

// Var dump within a <pre> tag
function pre_dump($var)
{
    echo "<pre>";
    var_dump($var);
    echo "</pre>";
}

// Connect to database posgresql00 twitter_media2
function pgx_00media2()
{
    static $conn = null;

    if ($conn !== null) {
        return $conn;
    }

    $connstr = 'host=postgresql00.mpi-sws.org
                dbname=twitter_media2
                user=twitter_media2
                password=fie0Wie7
                connect_timeout=10';

    $conn = pg_connect($connstr)
        or die('Could not connect: ' . pg_last_error());
    return $conn;
}

// Connect to database postgresql01 twitter_media
function pgx_01media()
{
    static $conn = null;

    if ($conn !== null) {
        return $conn;
    }

    $connstr = 'host=postgresql01.mpi-sws.org
                dbname=twitter_media
                user=twitter_media
                password=media_twitter
                connect_timeout=10';

    $conn = pg_connect($connstr)
        or die('Could not connect: ' . pg_last_error());
    return $conn;
}

// Connect to database postgresql00 twitter_data
function pgx_00data()
{
    static $conn = null;

    if ($conn !== null) {
        return $conn;
    }

    $connstr = 'host=postgresql00.mpi-sws.org
                dbname=twitter_data
                user=twitter_data
                password=twitter@mpi
                connect_timeout=10';

    $conn = pg_connect($connstr)
        or die('Could not connect: ' . pg_last_error());
    return $conn;
}

// Execute a query
function pgx_query($conn, $query)
{
    //echo $query;
    if (func_num_args() == 2) {
        $result = pg_query($conn, $query)
            or die('Query failed: ' . pg_last_error());
        return $result;
    }

    $args = func_get_args();
    //echo "<br><br>";

    $params = array_splice($args, 2);
    /*$string1=implode(",",$args);
    print_r($query."\n");
 foreach ($params as $key => $value) {
    echo $key . ' contains ' . $value . '<br/>';
}*/

    $result = pg_query_params($conn, $query, $params)

            or die('Query failed12542345: ' . pg_last_error());
    return $result;
}

// Log a mturk request
function mturk_access_log($workerid)
{
    $conn = pgx_00data();
    $sql = 'insert into mturk_sp_claim_nonpol_log_exp1 values ($1, $2, $3, $4, $5, $6)';
    pgx_query($conn, $sql,
              $workerid,
              $_SERVER["REMOTE_ADDR"],
              get_default($_SERVER, "HTTP_USER_AGENT", null),
              get_default($_SERVER, "HTTP_REFERER", null),
              $_SERVER["REQUEST_URI"],
              $_SERVER["REQUEST_TIME"]);
}

function linkify_text($raw_text)
{
    // create xhtml safe text (mostly to be safe of ampersands)
    $output = html_entity_decode($raw_text, ENT_NOQUOTES, 'UTF-8');
    $output = htmlentities($output, ENT_NOQUOTES, 'UTF-8');

    // parse urls
    $pattern     = '/([A-Za-z]+:\/\/[A-Za-z0-9-_]+\.[A-Za-z0-9-_:%&\?\/.=]+)/i';
    $replacement = '<a href="${1}" rel="external">${1}</a>';
    $output      = preg_replace($pattern, $replacement, $output);

    // parse the hashtags
    $pattern = '/#(\w+)\b/';
    $replacement = '<a href="https://twitter.com/search?q=%23${1}">#${1}</a>';
    $output = preg_replace($pattern, $replacement, $output);

    // parse the user mentions
    $pattern = '/@(\w+)\b/';
    $replacement = '<a href="https://twitter.com/${1}">@${1}</a>';
    $output = preg_replace($pattern, $replacement, $output);

    return $output;
}

function tweet_url($tweet_id, $sname)
{
    $tweet_url_out = sprintf("%s/status/%s", $sname,$tweet_id);
        return $tweet_url_out;
}


function survey_data_set($set_number)
{

#    $fname = "./amt_consensus_test_sets/seven_publishers_set_test.txt";
    // $fname = "./amt_consensus_test_sets/rumor_non-rumor_news_exp1.txt";
    $fname = "./amt_consensus_test_sets/snopes_nonpolitics_latest_20_news_per_lable_AMT_1.txt";
    #$fname = "./test_sets/set_".$set_number.".txt";
    $text = file_get_contents($fname);
    $sets1 = json_decode($text, true);

    #var_dump($sets1);
    return $sets1;

}

// Get the next tweet id
function mturk_next_id()
{
    $conn = pgx_00data();

    $sql = "select nextval('public.mturk_sp_claim_nonpol_seq_exp1')";
    $reply = pgx_query($conn, $sql);
    $row = pg_fetch_array($reply);
    return $row[0];
}

// Save mturk1 response

function survey_response($workerid, $tweet_id, $ra, $rb, $rc, $rd, $text)
//function survey_response($workerid, $tweet_id, $ra, $rb, $rc, $rd)
{
    $conn = pgx_00data();
    $sql = 'insert into mturk_sp_claim_nonpol_response_exp1 values ($1, $2, $3, $4, $5, $6, $7, $8)';
    pgx_query($conn, $sql, $workerid, $tweet_id, $ra, $rb, $rc,$_SERVER["REQUEST_TIME"] , $rd, $text);

}

function survey_demographics($worker_id, $nationality, $residence, $gender, $age, $degree, $employment, $income, $political_view, $race, $marital_status)
{
    $conn = pgx_00data();
    $sql = 'insert into mturk_sp_claim_nonpol_demographics1 values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11);';
    pgx_query($conn, $sql, $worker_id, $nationality, $residence, $gender, $age, $degree, $employment, $income, $political_view, $race, $marital_status);
    
}


function check_asnwered($workerid, $tweet_id)
{
    $conn = pgx_00data();
    $sql = 'select count(*) from mturk_sp_claim_nonpol_response_exp1 where workerid = $1 and tweet_id = $2';
    $reply = pgx_query($conn, $sql, $workerid, $tweet_id);
    $row = pg_fetch_array($reply);
    return (int)$row[0];
}

function get_answer($workerid, $tweet_id, $lastpage)
{
    $conn = pgx_00data();
        if ($lastpage==1){
    $sql = 'select ra, rb, rc, rd from mturk_sp_claim_nonpol_response_exp1 where workerid = $1 and tweet_id = $2';
        }
        else{
        $sql = 'select ra, rb, rc, text from mturk_sp_claim_nonpol_response_exp1 where workerid = $1 and tweet_id = $2';
        }
    $reply = pgx_query($conn, $sql, $workerid, $tweet_id);
    $row = pg_fetch_array($reply);
        if ($lastpage==1){
    return array((int)$row[0], (int)$row[1], (int)$row[2], (int)$row[3]);
        }
        else{
    return array((int)$row[0], (int)$row[1], (int)$row[2], (string)$row[3]);
        }
}



// Count proper responses by this worker
function survey_count_response($workerid)
{
    $conn = pgx_00data();
    $sql = 'select count(distinct (tweet_id))
            from mturk_sp_claim_nonpol_response_exp1
            where workerid = $1';
    $reply = pgx_query($conn, $sql, $workerid);
    $row = pg_fetch_array($reply);
    $total_responses = (int) $row[0];

    //$sql = 'select count(*) from mturk_crowd_signal_consensu_tweet_response_redblue_exp2 where workerid = $1 and ra = answer';
    //$reply = pgx_query($conn, $sql, $workerid);
    //$row = pg_fetch_array($reply);
    //$total_correct_responses = (int) $row[0];

    //return array($total_responses, $total_correct_responses);
        return array($total_responses);
}


function survey_count_response_test($workerid)
{
    $conn = pgx_00data();
    $sql = 'select count(distinct (tweet_id))
    from mturk_sp_claim_response_exp1
    where workerid = $1 and type = \'test\'';

    $reply = pgx_query($conn, $sql, $workerid);
    $row = pg_fetch_array($reply);
    $total_responses = (int) $row[0];

    $sql = 'select count(*) from mturk_sp_claim_nonpol_response_exp1 where workerid = $1 and ra = answer and type = \'test\'';
    $reply = pgx_query($conn, $sql, $workerid);
    $row = pg_fetch_array($reply);
    $total_correct_responses = (int) $row[0];

    return array($total_responses, $total_correct_responses);
}

