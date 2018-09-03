clear;
buildpath;
totalacc_induct = 0;
totalacc_transduct = 0;
for i=1:10
	[acc,acc2] = run_sun_split(i);
	totalacc_induct = totalacc_induct+acc;
    totalacc_transduct = totalacc_transduct+acc;
end
acc_induct = totalacc_induct/10;
acc_transduct = totalacc_transduct/10;
save('../result/SUN','acc_induct','acc_transduct');



