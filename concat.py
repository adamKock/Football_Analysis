



div = [ x*2 for x in range(1,11) if x%2==0]

dict_one ={"new york":"Sunny", "Cal": "Cold","London":"Cold","Auckland":"Mild"}

temp_in_c ={key:value for (key,value ) in dict_one.items() if  value=="Sunny"}








print(temp_in_c)