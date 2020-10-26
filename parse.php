<?php
    $input = $_POST['inputString'];
    $cmd = 'python "next_word_comparison_backoff.py" -i "'.$input.'"';
    $output = shell_exec($cmd);
    $json = json_decode($output,true);
    echo $output;
?>