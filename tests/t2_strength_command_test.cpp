// Exact lab-c public strength parser and roundtrip. used_by: native port receipt.
// rules: all finite nonnegative double values, no arbitrary cap, aliases and all modes.
#include "../NVEncCore/rgy_cmd.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
static void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
int main(){try{int positive=0,negative=0;
    for(const TCHAR* option:{_T("vpp-mcwiener"),_T("vpp-t2")})for(const TCHAR* mode:{_T("plain"),_T("box"),_T("full")})for(const TCHAR* strength:{_T(""),_T("0"),_T("0.25"),_T("1"),_T("2"),_T("1e300"),_T("1e-320"),_T("4.9406564584124654e-324")}){
        RGYParamVpp v,defaults;sArgsData args;tstring optionWithPrefix=tstring(_T("--"))+option,value=tstring(_T("mode="))+mode+_T(",profile=C:/profile with spaces/model.json,pp=SIM,iso=fixture");if(*strength)value+=tstring(_T(",strength="))+strength;const TCHAR* input[]={optionWithPrefix.c_str(),value.c_str()};int i=0;require(parse_one_vpp_option(option,input,i,2,&v,&args)==0,"valid strength accepted");double expected=*strength?std::strtod(tchar_to_string(strength).c_str(),nullptr):1.0;require(v.t2.strength==expected,"exact coefficient/default");auto cmd=gen_cmd(&v,&defaults,false);auto split=splitCommandLine(cmd.c_str());std::vector<const TCHAR*> argv;for(auto& item:split)if(!item.empty())argv.push_back(item.c_str());require(argv.size()==2,"quoted compound option");RGYParamVpp back;int j=0;require(parse_one_vpp_option(_T("vpp-mcwiener"),argv.data(),j,int(argv.size()),&back,&args)==0,"strength roundtrip parses");require(v.t2==back.t2,"roundtrip preserves exact double");++positive;
    }
    for(const TCHAR* invalid:{_T("-1"),_T("nan"),_T("inf"),_T("-inf"),_T("1e309"),_T("1junk"),_T("")}){RGYParamVpp v;sArgsData args;tstring value=tstring(_T("mode=box,profile=a,pp=x,iso=y,strength="))+invalid;const TCHAR* argv[]={_T("--vpp-mcwiener"),value.c_str()};int i=0;require(parse_one_vpp_option(_T("vpp-mcwiener"),argv,i,2,&v,&args)!=0,"invalid strength rejected");++negative;}
    std::cout<<"{\"result\":\"PASS\",\"roundtrip_cases\":"<<positive<<",\"invalid_cases\":"<<negative<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
