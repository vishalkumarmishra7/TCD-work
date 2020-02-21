
int cnt;
Table table;
float s_width = 2000;
float s_height = 1000;
float[] lonc = new float[20];
float[] latc = new float[20];
String[] city = new String[20];
float[] lont = new float[9];
float[] temp = new float[9];
float[] days = new float[9];
String[] mon = new String[9];
float[] day = new float[9];
float[] lonp = new float[48];
float[] latp = new float[48];
float[] surv = new float[48];
String[] dir = new String[48];
float[] div = new float[48];
int step;
String state;
PImage webImg;
String flag  = "A";

void setup()
{
    /***************Data read******************/
    table = loadTable("C:\\Users\\adhis\\Desktop\\Main Study Stuff\\7. Data Visualization\\Assignments\\A1.2\\Minard\\minard-data-1.csv", "header");
    cnt = 1;
    for (TableRow row : table.rows()) 
    {
        lonc[cnt-1] = row.getFloat("LONC");
        latc[cnt-1] = row.getFloat("LATC");
        city[cnt-1] = row.getString("CITY");
        cnt += 1; 
    }
    
    table = loadTable("C:\\Users\\adhis\\Desktop\\Main Study Stuff\\7. Data Visualization\\Assignments\\A1.2\\Minard\\minard-data-2.csv", "header");
    cnt = 1;
    for (TableRow row : table.rows()) 
    {
        lont[cnt-1] = row.getFloat("LONT");
        temp[cnt-1] = row.getFloat("TEMP");
        days[cnt-1] = row.getFloat("DAYS");
        mon[cnt-1] = row.getString("MON");
        day[cnt-1] = row.getFloat("DAY");
        cnt += 1; 
    }
    
    table = loadTable("C:\\Users\\adhis\\Desktop\\Main Study Stuff\\7. Data Visualization\\Assignments\\A1.2\\Minard\\minard-data-3.csv", "header");
    cnt = 1;
    for (TableRow row : table.rows()) 
    {
        lonp[cnt-1] = row.getFloat("LONP");
        latp[cnt-1] = row.getFloat("LATP");
        surv[cnt-1] = row.getFloat("SURV");
        dir[cnt-1] = row.getString("DIR");
        div[cnt-1] = row.getFloat("DIV");
        //print(lonp[cnt-1] + "     " + latp[cnt-1] + "     " + surv[cnt-1] + "     " + dir[cnt-1] + "     " + div[cnt-1] + '\n');
        cnt += 1; 
    }
    
    
    /**************Setting up display***********/
    size(2400, 2000);
    background(255);
    //noStroke();
    noLoop();
    frameRate(5);
}


void c_line(float x1, float y1, float x2, float y2, float weight, String dir)
{
    strokeWeight(weight);
    strokeJoin(MITER);
    if(dir.equals("A"))
    {
        stroke(240, 86, 55, 150);
    }
    else
    {
        stroke(50, 117, 240, 150);
    }
    line(x1+200, height - y1 - 850, x2 + 200, height - y2 - 850);
}

void temp(float x1, float y1, float x2, float y2, float temp)
{
    strokeWeight(10);
    strokeJoin(MITER);
    stroke(98, 252, 134, 150);
    line(x1+200,height - y1 - 130,x2+200,height - y2 - 130);
    fill(0);
    textSize(30);
    text(temp,x1+200,height - y1 - 140);
}

void city(float x, float y, String t)
{
    fill(0);
    textSize(30);
    text(t, x+150, height - y - 850);
}

void draw_path(int s)
{
    for(int i = 0; i < s - 1; i++)
    {
        if(div[i] == div[i+1])
        {
            c_line(((lonp[i]-min(lonp))*s_width)/(max(lonp)-min(lonp)), ((latp[i]-min(latp))*s_height)/(max(latp)-min(latp)), ((lonp[i+1]-min(lonp))*s_width)/(max(lonp)-min(lonp)), ((latp[i+1]-min(latp))*s_height)/(max(latp)-min(latp)), (surv[i]*70)/max(surv), dir[i]);
        
            if(dir[i].equals("A"))
            {
                state = "attacking";
            }
            else
            {
                state = "retreating";
            }    
        }
        
    }
    fill(0,0,0,200);
    textSize(50);
    if(state != null)
    {
        text("Division "+div[s-1]+" "+state,4*width/5 - 200,height/2);
    }
    
    fill(0,0,0,200);
    textSize(60);
    String t = "Minard's depiction of Napoleon's army's march to Moscow";
    text(t,width/2 - textWidth(t)/2,80);
}

void draw_static()
{
    for(int i = 0; i < lont.length - 1; i++)
    {
        temp(((lont[i]-min(lont))*s_width)/(max(lont)-min(lont)),((temp[i]-min(temp))*(s_height-300))/(max(temp)-min(temp)),((lont[i+1]-min(lont))*s_width)/(max(lont)-min(lont)),((temp[i+1]-min(temp))*(s_height-300))/(max(temp)-min(temp)),temp[i]);
    }
    
    for(int i = 0; i < lonc.length; i++)
    {
        city(((lonc[i]-min(lonc))*s_width)/(max(lonc)-min(lonc)),((latc[i]-min(latc))*s_height)/(max(latc)-min(latc)),city[i]);
    }
    
    int w = 100;
    while(w <= width - 100)
    {
        stroke(0, 20);
        strokeWeight(3);
        line(w, 100, w, height - 100);
        w += 50;
    }
    
    int h = 100;
    while(h <= height - 100)
    {
        stroke(0, 20);
        strokeWeight(3);
        line(100, h, width - 100, h);
        h += 50;
    }
    
    textSize(35);
    fill(240, 86, 55, 150);
    rect(1600, 1700, 50, 50);
    
    fill(50, 117, 240, 150);
    rect(1600, 1800, 50, 50);
    
    fill(98, 252, 134, 150);
    rect(1600, 1600, 50, 50);
    
    fill(0);
    text("Attacking Army Trajectory",1675,1735);
    text("Retreating Army Trajectory",1675,1835);
    text("Temperature",1675,1635);
}

void draw()
{
    background(255);
    
    step += 1;
    
    if(step > lonp.length)
    {
        step = 1;
        flag = "A";
    }
    if(dir[step-1].equals("R"))
    {
        flag = "R";
    }
    
    //tint(255,255,255,50);
    
    //if(flag.equals("A"))
    //{
    //    webImg = loadImage("C:\\Users\\adhis\\Desktop\\Main Study Stuff\\7. Data Visualization\\Assignments\\A1.2\\Minard\\a.jpg");
    //    image(webImg, 0, 0, width, height);
    //}
    //else
    //{
    //    webImg = loadImage("C:\\Users\\adhis\\Desktop\\Main Study Stuff\\7. Data Visualization\\Assignments\\A1.2\\Minard\\r.jpg");
    //    image(webImg, 0, 0, width, height);
    //}
    
    draw_path(step);
    
    draw_static();
    
    saveFrame("C:\\Users\\adhis\\Desktop\\Main Study Stuff\\7. Data Visualization\\Assignments\\A1.2\\Minard\\frames\\"+frameCount+".jpg");
}

void keyPressed()
{
    redraw();
}

//void keyPressed()
//{
//    if(key == 97)
//    {
//        step += 1;
//        if(step > lonp.length*10)
//        {
//            step = 0;
//        }
//        redraw();
//    }
//}