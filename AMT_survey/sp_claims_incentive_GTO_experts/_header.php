<!DOCTYPE html>
<html lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html;charset=utf-8" />

<title><?= TITLE ?></title>
<link href="css/bootstrap.min.css" rel="stylesheet" media="screen">
<link href="css/common.css" rel="stylesheet" media="screen">

<script src="js/jquery-1.8.2.min.js"></script>
<script src="js/bootstrap.min.js"></script>

<script type="text/javascript">

function callNext(p, w, ra, rb, rc, rd, text)
{
    
    window.location.href = "survey.php?p=" + p + "&w=" + w + "&ra=" + ra + "&rb=" + rb + "&rc=" + rc + "&rd=" + rd + "&text=" + text;
}

function callLast(p, w, demographic_nationality, demographic_residence, demographic_gender, demographic_age,  demographic_degree, demographic_employment, demographic_income, demographic_political_view , demographic_race, demographic_marital_status) {
    window.location.href = "survey.php?p=" + p + "&w=" + w + "&demographic_nationality=" + demographic_nationality + "&demographic_residence=" + demographic_residence + "&demographic_gender=" + demographic_gender + "&demographic_age=" + demographic_age + "&demographic_degree=" + demographic_degree + "&demographic_employment=" + demographic_employment + "&demographic_income=" + demographic_income + "&demographic_political_view=" + demographic_political_view  + "&demographic_race=" + demographic_race + "&demographic_marital_status=" + demographic_marital_status;
}


</script>

<style>

input[type="range"]{
    -webkit-appearance: none;
    -moz-apperance: none;
    width: 100px;
    height: 4px;
    padding: 0px;

    background-image:
     -webkit-linear-gradient(left,rgba(76,171,217,1), rgba(200,73,55,1));
    border-radius: 2px;
    margin-top: 25px;
}

input[type="range"]::-webkit-slider-thumb{
    -webkit-appearance:none;
    -moz-apperance:none;
    width:3px;
    height:12px;
    background: #737373;
    z-index: 22;
}

</style>



</head>
<body>
<div class="container">
