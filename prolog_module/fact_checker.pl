:- module(fact_checker, [is_fake/1]).

is_fake(Text) :- contains(Text, "flat earth").
is_fake(Text) :- contains(Text, "alien").
is_fake(Text) :- contains(Text, "cure for cancer").
is_fake(Text) :- contains(Text, "conspiracy").

contains(Text, Sub) :- sub_string(Text, _, _, _, Sub).