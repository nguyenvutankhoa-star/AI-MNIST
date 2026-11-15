/* Standalone C11, MNIST HWIO/NHWC. Run from repository root.
 * gcc -O3 -std=c11 "Source C/mnist_PTQ.c" -lm -o artifacts/ptq.exe
 * Usage: executable [epochs=1(QAT)/0(PTQ)] [train_count=60000] [test_count=10000]
 * Inputs: weights directory text files; artifacts/train_images.bin (float32 [0,1]),
 * artifacts/train_labels.bin (int32); data/test_{images,labels}.bin.
 * Frozen BN AFTER ReLU: fold a=gamma/sqrt(var+.001), t=beta-a*mean:
 * W'=aW, b'=ab+t, y=max(W'x+b',t) if a>=0 else min(...,t).
 * Thus this is NOT an ordinary Conv-BN-ReLU rewrite.
 * Deployment: signed INT8 weights/activations, INT32 MAC, power-of-two
 * per-tensor scales, INT64 requantization intermediate; no float in infer_int.
 * QAT: all-layer SGD, fake quant on weights/activations, clipped STE,
 * frozen BN and scales. No test labels/images used for training/calibration.
 * Binary exports are little-endian on tested Windows host; see report format.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#define QAT 0
#define TAG "PTQ"
typedef struct { int in,ci,co,h,p,n; float w[4000],b[16],t[16];
 int sign[16],ew,ei,eo; int8_t qw[4000]; int32_t qb[16],qt[16]; } Layer;
static Layer l[3];
static float f1[1352],f2[400],z1[5408],z2[1936],input[784];
static int ix1[1352],ix2[400];
static int8_t qx[784],q1[1352],q2[400];
static void fail(const char *s) {fprintf(stderr,"ERROR: %s\n",s);exit(1);}
static FILE *openf(const char *p,const char *m){FILE*f=fopen(p,m);if(!f)fail(p);return f;}
static void load(const char *name,float *a,int n){char p[256];snprintf(p,sizeof p,"weights/%s.txt",name);FILE*f=openf(p,"r");
 for(int i=0;i<n;i++){int c;do{c=fgetc(f);}while(c==','||c==' '||c=='\n'||c=='\r'||c=='\t');if(c==EOF)fail(p);ungetc(c,f);if(fscanf(f,"%f",a+i)!=1||!isfinite(a[i]))fail(p);}
 int c;while((c=fgetc(f))!=EOF)if(c!=','&&c!=' '&&c!='\n'&&c!='\r'&&c!='\t')fail("extra parameter");fclose(f);}
static void init(void){
 l[0].in=28;l[0].ci=1;l[0].co=8;l[0].h=26;l[0].p=13;l[0].n=72;
 l[1].in=13;l[1].ci=8;l[1].co=16;l[1].h=11;l[1].p=5;l[1].n=1152;
 l[2].ci=400;l[2].co=10;l[2].n=4000;
 for(int j=0;j<3;j++){Layer*a=l+j;char s[80];snprintf(s,sizeof s,j<2?"conv%d_weights":"dense_weights",j+1);load(s,a->w,a->n);
 snprintf(s,sizeof s,j<2?"conv%d_bias":"dense_bias",j+1);load(s,a->b,a->co);
 if(j<2){float g[16],b[16],m[16],v[16];snprintf(s,sizeof s,"bn%d_gamma",j+1);load(s,g,a->co);snprintf(s,sizeof s,"bn%d_beta",j+1);load(s,b,a->co);snprintf(s,sizeof s,"bn%d_mean",j+1);load(s,m,a->co);snprintf(s,sizeof s,"bn%d_var",j+1);load(s,v,a->co);
 for(int c=0;c<a->co;c++){if(v[c]<0)fail("negative BN variance");float sc=g[c]/sqrtf(v[c]+.001f);a->t[c]=b[c]-sc*m[c];a->sign[c]=sc>=0?1:-1;a->b[c]=sc*a->b[c]+a->t[c];for(int k=c;k<a->n;k+=a->co)a->w[k]*=sc;}}
 }}
static int exponent(float mx){return mx>0?(int)ceil(log2((double)mx/127.0)):-20;}
static int quant(float x,int e){double v=round(ldexp((double)x,-e));return v>127?127:v< -127?-127:(int)v;}
static float fake(float x,int e){return ldexpf((float)quant(x,e),e);}
static void pack(void){for(int j=0;j<3;j++){Layer*a=l+j;for(int k=0;k<a->n;k++)a->qw[k]=(int8_t)quant(a->w[k],a->ew);
 for(int c=0;c<a->co;c++){double b=round(ldexp((double)a->b[c],-a->ei-a->ew));double t=round(ldexp((double)a->t[c],-a->ei-a->ew));if(fabs(b)>INT32_MAX-20000000||fabs(t)>INT32_MAX-20000000)fail("bias overflow");a->qb[c]=(int32_t)b;a->qt[c]=(int32_t)t;}}}
static float weight(Layer*a,int k,int fq){return fq?ldexpf(a->qw[k],a->ew):a->w[k];}
static void conv_float(Layer*a,const float*x,float*y,float*z,int*ix,int fq){
 for(int i=0;i<a->h;i++)for(int j=0;j<a->h;j++)for(int c=0;c<a->co;c++){
 float s=a->b[c];for(int u=0;u<3;u++)for(int v=0;v<3;v++)for(int k=0;k<a->ci;k++)s+=x[((i+u)*a->in+j+v)*a->ci+k]*weight(a,((u*3+v)*a->ci+k)*a->co+c,fq);
 z[(i*a->h+j)*a->co+c]=s;}
 for(int i=0;i<a->p;i++)for(int j=0;j<a->p;j++)for(int c=0;c<a->co;c++){float best=-INFINITY;int idx=0;
 for(int u=0;u<2;u++)for(int v=0;v<2;v++){int k=((2*i+u)*a->h+2*j+v)*a->co+c;float s=z[k];s=a->sign[c]>0?fmaxf(s,a->t[c]):fminf(s,a->t[c]);if(s>best){best=s;idx=k;}}
 int o=(i*a->p+j)*a->co+c;ix[o]=idx;y[o]=fq?fake(best,a->eo):best;}}
static void forward(const float*x,float*out,int fq){for(int k=0;k<784;k++)input[k]=fq?fake(x[k],l[0].ei):x[k];
 conv_float(l,input,f1,z1,ix1,fq);conv_float(l+1,f1,f2,z2,ix2,fq);
 for(int c=0;c<10;c++){float s=l[2].b[c];for(int k=0;k<400;k++)s+=f2[k]*weight(l+2,k*10+c,fq);out[c]=s;}}
/* Symmetric round-to-nearest, ties away from zero; no signed right shift. */
static int32_t rescale(int32_t x,int e){int64_t v=x;if(e>=0){if(e>30)fail("invalid scale");v*=INT64_C(1)<<e;}else{int s=-e;if(s>60)fail("invalid scale");int64_t d=INT64_C(1)<<s;v=v>=0?(v+d/2)/d:-((-v+d/2)/d);}return v>127?127:v< -127?-127:(int32_t)v;}
static void conv_int(const Layer*a,const int8_t*x,int8_t*y){
 /* Fuse conv, threshold activation and pool: no full convolution feature map. */
 for(int i=0;i<a->p;i++)for(int j=0;j<a->p;j++)for(int c=0;c<a->co;c++){int best=-127;
 for(int u=0;u<2;u++)for(int v=0;v<2;v++){int32_t s=a->qb[c];for(int r=0;r<3;r++)for(int t=0;t<3;t++)for(int k=0;k<a->ci;k++)s+=(int32_t)x[((2*i+u+r)*a->in+2*j+v+t)*a->ci+k]*a->qw[((r*3+t)*a->ci+k)*a->co+c];
 if(a->sign[c]>0){if(s<a->qt[c])s=a->qt[c];}else if(s>a->qt[c])s=a->qt[c];int q=rescale(s,a->ei+a->ew-a->eo);if(q>best)best=q;}
 y[(i*a->p+j)*a->co+c]=(int8_t)best;}}
static int infer_int(const int8_t*x){conv_int(l,x,q1);conv_int(l+1,q1,q2);int best=0;int32_t mx=INT32_MIN;
 for(int c=0;c<10;c++){int32_t s=l[2].qb[c];for(int k=0;k<400;k++)s+=(int32_t)q2[k]*l[2].qw[k*10+c];if(s>mx){mx=s;best=c;}}return best;}
static int argmax(const float*x){int b=0;for(int c=1;c<10;c++)if(x[c]>x[b])b=c;return b;}
static float *images(const char*p,int*n){FILE*f=openf(p,"rb");if(fseek(f,0,SEEK_END))fail(p);long z=ftell(f);if(z<=0||z%(784*4))fail("invalid image size");rewind(f);float*x=malloc((size_t)z);if(!x||fread(x,1,(size_t)z,f)!=(size_t)z)fail(p);fclose(f);*n=(int)(z/(784*4));for(long k=0;k<z/4;k++)if(!isfinite(x[k])||x[k]<0||x[k]>1)fail("pixels must be float32 [0,1]");return x;}
static int32_t *labels(const char*p,int n){FILE*f=openf(p,"rb");int32_t*y=malloc((size_t)n*4);if(!y||fread(y,4,n,f)!=(size_t)n||fgetc(f)!=EOF)fail("label size");fclose(f);for(int k=0;k<n;k++)if(y[k]<0||y[k]>9)fail("label range");return y;}
static void calibrate(const float*x,int n){float mx[2]={0,0},o[10];for(int i=0;i<n;i++){forward(x+i*784,o,0);for(int k=0;k<1352;k++)mx[0]=fmaxf(mx[0],fabsf(f1[k]));for(int k=0;k<400;k++)mx[1]=fmaxf(mx[1],fabsf(f2[k]));}
 l[0].ei=exponent(1);l[0].eo=l[1].ei=exponent(mx[0]);l[1].eo=l[2].ei=exponent(mx[1]);
 for(int j=0;j<3;j++){float m=0;for(int k=0;k<l[j].n;k++)m=fmaxf(m,fabsf(l[j].w[k]));l[j].ew=exponent(m);}pack();}
static int ste(float v,int e){return fabsf(v)<=ldexpf(127,e);}
static void backward_conv(Layer*a,const float*x,const float*z,const int*ix,const float*dy,float*dx,float lr){
 float dw[1152]={0},db[16]={0};memset(dx,0,(size_t)a->in*a->in*a->ci*sizeof(float));
 for(int o=0;o<a->p*a->p*a->co;o++){int c=o%a->co,k=ix[o],i=k/a->co/a->h,j=k/a->co%a->h;float v=z[k];float act=a->sign[c]>0?fmaxf(v,a->t[c]):fminf(v,a->t[c]);
 if(!ste(act,a->eo)||(a->sign[c]>0?v<=a->t[c]:v>=a->t[c]))continue;
 float g=dy[o];db[c]+=g;
 for(int r=0;r<3;r++)for(int t=0;t<3;t++)for(int ci=0;ci<a->ci;ci++){int wi=((r*3+t)*a->ci+ci)*a->co+c,xi=((i+r)*a->in+j+t)*a->ci+ci;dw[wi]+=g*x[xi];dx[xi]+=g*ldexpf(a->qw[wi],a->ew);}}
 for(int k=0;k<a->n;k++)if(ste(a->w[k],a->ew))a->w[k]-=lr*dw[k];
 for(int c=0;c<a->co;c++)a->b[c]-=lr*db[c];}
static float train_one(const float*x,int label,float lr){float o[10],g[10],d2[400]={0},d1[1352],d0[784];forward(x,o,1);float mx=o[argmax(o)],sum=0;for(int c=0;c<10;c++){g[c]=expf(o[c]-mx);sum+=g[c];}float loss=-logf(fmaxf(g[label]/sum,1e-30f));for(int c=0;c<10;c++)g[c]=g[c]/sum-(c==label);
 for(int k=0;k<400;k++)for(int c=0;c<10;c++){int wi=k*10+c;d2[k]+=g[c]*ldexpf(l[2].qw[wi],l[2].ew);if(ste(l[2].w[wi],l[2].ew))l[2].w[wi]-=lr*g[c]*f2[k];}
 for(int c=0;c<10;c++)l[2].b[c]-=lr*g[c];
 backward_conv(l+1,f1,z2,ix2,d2,d1,lr);backward_conv(l,input,z1,ix1,d1,d0,lr);pack();return loss;}
static void put(FILE*f,const void*p,size_t s,size_t n){if(fwrite(p,s,n,f)!=n)fail("export write");}
static void export_model(void){char path[128];snprintf(path,sizeof path,"artifacts/%s_fpga.bin",TAG);FILE*f=openf(path,"wb");put(f,"MNI8",1,4);int32_t ver=1;put(f,&ver,4,1);
 for(int j=0;j<3;j++){Layer*a=l+j;int32_t meta[8]={a->in,a->ci,a->co,a->p,a->n,a->ei,a->ew,a->eo};put(f,meta,4,8);put(f,a->qw,1,a->n);put(f,a->qb,4,a->co);put(f,a->qt,4,a->co);for(int c=0;c<a->co;c++){int32_t s=a->sign[c];put(f,&s,4,1);}}if(fclose(f))fail("export close");}
static int number(const char*s){char*end;long n=strtol(s,&end,10);if(!*s||*end||n<0||n>1000000)fail("invalid argument");return(int)n;}
int main(int argc,char**argv){if(sizeof(float)!=4||sizeof(int32_t)!=4)fail("32-bit file types required");int ep=argc>1?number(argv[1]):QAT,nt=argc>2?number(argv[2]):60000,ne=argc>3?number(argv[3]):10000;if(argc>4||(!QAT&&ep)||!nt||!ne)fail("usage: program epochs train_count test_count");
 init();int ntr,nte;float*tr=images("artifacts/train_images.bin",&ntr),*te=images("data/test_images.bin",&nte);int32_t*yt=labels("artifacts/train_labels.bin",ntr),*ye=labels("data/test_labels.bin",nte);if(nt>ntr||ne>nte)fail("sample count exceeds data");
 calibrate(tr,nt<512?nt:512);double start=(double)clock()/CLOCKS_PER_SEC;
 for(int e=0;e<ep;e++){double loss=0;for(int k=0;k<nt;k++)loss+=train_one(tr+k*784,yt[k],.0001f);printf("epoch=%d samples=%d loss=%.6f\n",e+1,nt,loss/nt);fflush(stdout);}printf("training_seconds=%.3f\n",(double)clock()/CLOCKS_PER_SEC-start);
 char pp[128];snprintf(pp,sizeof pp,"artifacts/%s_predictions.bin",TAG);FILE*pred=openf(pp,"wb");
 int correct=0,fp=0,agree=0;double ti=0,tf=0;for(int k=0;k<ne;k++){float o[10];double s=(double)clock()/CLOCKS_PER_SEC;forward(te+k*784,o,0);int b=argmax(o);tf+=(double)clock()/CLOCKS_PER_SEC-s;for(int i=0;i<784;i++)qx[i]=(int8_t)quant(te[k*784+i],l[0].ei);s=(double)clock()/CLOCKS_PER_SEC;int q=infer_int(qx);ti+=(double)clock()/CLOCKS_PER_SEC-s;int32_t pair[2]={b,q};put(pred,pair,4,2);correct+=q==ye[k];fp+=b==ye[k];agree+=q==b;}
 if(fclose(pred))fail("predictions close");
 export_model();printf("%s test=%d int8_correct=%d accuracy=%.4f%% float_correct=%d float_accuracy=%.4f%% agreement=%.4f%% int_seconds=%.3f float_seconds=%.3f\n",TAG,ne,correct,100.0*correct/ne,fp,100.0*fp/ne,100.0*agree/ne,ti,tf);
 printf("weight_bytes=5224 activation_workspace_bytes=2536 export=artifacts/%s_fpga.bin\n",TAG);free(tr);free(te);free(yt);free(ye);return 0;}
