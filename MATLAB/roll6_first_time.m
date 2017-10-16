clc
clear all
close all

N = 10000;
accepted = 0;
match = 0;

cumcnt = 0;
for i = 1:N
    cnt = 0;
    y = 0;
    while(y ~= 6)
        y = randint(1,1,[0,2]);
        y = 2 + 2 * y;
        cnt = cnt + 1;
    end
    cumcnt = cumcnt + cnt;
end

mean = cumcnt / N;