import java.util.Random;

Random rand = new Random();
Table table;
String[] l_month = new String[12];
float[] l_zd = new float[12];
float[] l_wi = new float[12];
float[] l_aoc = new float[12];
int cnt = 0;
double d;
int sf = 25;
String start_month = "Jan 1855";
float lastAngle;
float la = -3; 
float sf2 = 10;
int ra = 60; //rand.nextInt(180);

void setup() 
  {
    /***************Data read******************/
    size(2500,1500);
    background(0);
    table = loadTable("C:\\Users\\adhis\\Desktop\\Main Study Stuff\\7. Data Visualization\\Assignments\\A1.2\\Nightingale's Rose\\Rose chart final data.csv", "header");
    for (TableRow row : table.rows()) 
      {
        String month = row.getString("Month");
        if (month.equals(start_month)) 
          {
            print("Found month \n");
            cnt = 1;
          }
        if (cnt > 0 & cnt < 13) 
          {
            l_month[cnt-1] = month;
            float zd = row.getFloat("Zymotic diseases");
            l_zd[cnt-1] = zd;
            float wi = row.getFloat("Wounds & injuries");
            l_wi[cnt-1] = wi;
            float aoc = row.getFloat("All other causes");
            l_aoc[cnt-1] = aoc;
            cnt += 1;
          }
      }
    
    /**************Setting up display***********/
    //noStroke();
    noLoop();  // Run once and stop
    //frameRate(1);
  }
  
void pieChart(String[] l_month, float[] l, int r, int g, int b, int op, float lastAngle) 
  {
    float tf = 10;
    float tf2 = 7.5;
    int n = rand.nextInt(10);
    float lastAngle2 = lastAngle + ra;
    for (int i = 0; i < l_month.length; i++) 
      {
        //float col = map(i, 0, l_month.length, 0, 255);
        d = Math.pow(l[i]/3.1415,0.5)*2*sf;
        float df = (float)d;
        float df2 = (float)d/0.7;
        if (op == 255)
          {
            float theta = lastAngle + radians(360/24);
      float theta2 = lastAngle2 + radians(360/24);
            fill(0);
            textSize(22.5);
            text(l_month[i].substring(0,3).toUpperCase(), width/4 - 150 + (float)Math.cos(theta)*(max(df,(float)Math.pow(max(l)*0.3/3.1415,0.5)*2*sf*sf2/tf)+80)/2, height/2 + (float)Math.sin(theta)*(max(df,(float)Math.pow(max(l)*0.3/3.1415,0.5)*2*sf*sf2/tf)+80)/2);
            text(l_month[i].substring(0,3).toUpperCase(), 3*width/4 + (float)Math.cos(theta2)*(max(df2,(float)Math.pow(max(l)*0.3/3.1415,0.5)*2*sf*sf2/tf2)+80)/2 + 100, height/2 + (float)Math.sin(theta2)*(max(df2,(float)Math.pow(max(l)*0.3/3.1415,0.5)*2*sf*sf2/tf2)+80)/2);
          }
        fill(r,g+n,b,op);
        arc(width/4 -150, height/2, df*sf2/tf, df*sf2/tf, lastAngle, lastAngle+radians(360/12));
        arc(3*width/4 + 100, height/2, df*sf2/tf2, df*sf2/tf2, lastAngle2, lastAngle2+radians((360)/12));
        lastAngle += radians(360/12);
        lastAngle2 += radians(360/12);
      }
  }
  
void d1(float la) 
  {
    
    //background(255);
    
    int w = 100;
    while(w <= width - 100)
    {
        stroke(0, 20);
        strokeWeight(3);
        line(w, 150, w, height - 100);
        w += 50;
    }
    
    int h = 150;
    while(h <= height - 100)
    {
        stroke(0, 20);
        strokeWeight(3);
        line(100, h, width - 100, h);
        h += 50;
    }
    
    lastAngle = la/12;
    pieChart(l_month, l_zd, 200, 60, 100, 255, lastAngle);
    pieChart(l_month, l_wi, 170, 70, 200, 225, lastAngle);
    pieChart(l_month, l_aoc, 70, 200, 90, 200, lastAngle);
    
    textSize(30);
    
    fill(200, 60, 100, 255);
    rect(width/2 - 200, height - 200, 50, 50);
    fill(170, 70, 200, 200);
    rect(width/2 - 200, height - 300, 50, 50);
    fill(70, 200, 90, 150);
    rect(width/2 - 200, height - 400, 50, 50);
    
    fill(0);
    text("Zymotic diseases", width/2 - 130, height - 160);
    text("Wounds & injuries", width/2 - 130, height - 260);
    text("All other causes", width/2 - 130, height - 360);
  }

//Clock Animation (needs looping)
//void draw()
//{
//    background(255);
//    la += 1;
//    redraw();
//    d1(la);
//}

//Key Press Clock Animation Looping needs to be disabled
void draw()
{
    la += 1;
    background(255,150);
    //tint(255,255,255,150);
    //PImage img = loadImage("C:\\Users\\adhis\\Desktop\\Main Study Stuff\\7. Data Visualization\\Assignments\\A1.2\\Nightingale's Rose\\bg.jpg");
    //image(img, 0, 0, width, height);
    d1(la);
    
    textSize(70);
    fill(0);
    String t = "Diagram of the Causes of Mortality in the Army in the East";
    text(t,width/2 - textWidth(t)/2,100);
    
    saveFrame("C:\\Users\\adhis\\Desktop\\Main Study Stuff\\7. Data Visualization\\Assignments\\A1.2\\Nightingale's Rose\\frames\\" + frameCount + ".jpg");
}

void keyPressed() { 
  if(key == 97)
    {
        redraw();
    }
}

//void draw()
//{
//    sf2 += 1;
//    if(sf2>10)
//    {
//        sf2 = 5;
//    }
//    background(255);
//    d1(la);
//}