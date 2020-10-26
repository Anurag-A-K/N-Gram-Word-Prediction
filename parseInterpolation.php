<?php
    $input = $_POST['inputString'];
    ini_set('max_execution_time', 300);
    $cmd = 'python "next_word_comparison_interpolation.py" -i "'.$input.'"';
    $output = shell_exec($cmd);
    $json = json_decode($output,true);
    echo $output;
?>