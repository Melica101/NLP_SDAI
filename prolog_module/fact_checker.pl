:- module(fact_checker, [is_fake/1, is_real/1]).

fake_news_terms([
    "flat earth",
    "alien",
    "UFO",
    "ghost",
    "cure for cancer",
    "miracle",
    "conspiracy",
    "scandal",
    "secret", "sex",
    "fake", "conspiracy", "hoax", "scam", "fraud"
]).

real_news_terms([
    "CNN",
    "BBC News",
    "Reuters",
    "Associated Press",
    "CNN Report", "Fox News Alert",
    "rule", "policy", "regulation", "law",
    "government", "election", "vote"
]).

is_fake(Text) :-
    fake_news_terms(FakeTerms),
    member(Sub, FakeTerms),
    contains(Text, Sub).

is_real(Text) :-
    real_news_terms(RealTerms),
    member(Sub, RealTerms),
    contains(Text, Sub).

contains(Text, Sub) :- sub_string(Text, _, _, _, Sub).